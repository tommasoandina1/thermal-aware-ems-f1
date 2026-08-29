"""
Safety layer for the EMS environment.

Implements the "safe exploration" pattern of Garcia & Fernandez / Brunke et al.:
an action proposed by the agent is projected onto the nearest point of the set
of *physically and regulatorily admissible* actions, computed analytically from
the current state. The agent therefore cannot emit an action that violates a
Level-III (hard) constraint, and those constraints never have to be learnt.

The set is expressed in the electrical domain (battery terminal power P2 [W]),
which is the same domain in which the regulatory 350 kW limit and the per-lap
deploy budget are defined. Sign convention inherited from plant/battery.py:

    P2 > 0  -> discharge (deploy / boost)
    P2 < 0  -> charge (regeneration or ICE-driven charging)

Five constraints are enforced simultaneously:

    1. MGU-K instantaneous envelope, thermally derated:
                                          P2 <= P_MGU_max * derate(T_bat)
    2. Equivalent-circuit solvability:    P2 <= Uoc^2 / (4 R_int)
    3. Per-lap deploy energy budget:      P2 * dt <= E_deploy_max - E_deploy_acc
    4. SoC window:                        SoC_min <= SoC_next <= SoC_max
    5. Physical availability of braking:  P_mech >= P_gb when P_gb < 0

Constraint 2 is the one the plant would otherwise apply *silently* inside
battery_step(); enforcing it here instead means the requested and delivered
powers stay consistent and the shortfall reported by powertrain() remains the
single source of truth.

A note on constraint 2. Uoc^2/(4 R_int) is the maximum-power-transfer point of
the equivalent circuit, not a physical operating limit: it corresponds to half
the open-circuit voltage dropped across R_int, roughly 15 kA at 300 V, and it
sits at 1.8-3.0 MW across the SoC window. It is included here only so that the
admissible interval can never contain a command for which the terminal-voltage
quadratic has no real solution. The thermal limit that actually binds is
constraint 1, on the MGU-K deploy envelope, matching plant/powertrain.MGU_K().
"""

from __future__ import annotations
import numpy as np

from plant.battery import Uoc, thermal_derating_factor


def feasible_P2_bounds(SoC, T_bat, E_deploy_acc, P_gb, params, dt, coef,
                       allow_ice_charging: bool = True):
    """
    Analytic lower/upper bound on the admissible battery terminal power.

    Returns
    -------
    (lb, ub) : tuple of float
        Admissible interval for P2 [W]. Always satisfies lb <= ub: if the
        constraints are mutually infeasible (which can only happen at the very
        edge of the SoC window), the interval collapses to a single point
        rather than producing an inverted interval. An inverted interval fed to
        np.linspace or to a uniform sampler is silently meaningless, which is
        exactly the failure mode that has to be excluded by construction.
    """
    R_int = params['R_int']
    Q_bat = params['E_pack_capacity'] / params['V_oc_nom']      # [C]
    eta_MGU = params['eta_MGU']

    U = float(Uoc(SoC, coef))
    derate = thermal_derating_factor(T_bat, params)

    # --- 1. component envelope, thermally derated on the deploy side ------
    ub = float(params['P_MGU_max']) * derate
    lb = float(params['P_MGU_min'])

    # --- 2. equivalent-circuit solvability (not a physical limit) ---------
    ub = min(ub, U ** 2 / (4.0 * R_int))

    # --- 3. per-lap deploy budget -----------------------------------------
    e_left = max(0.0, params['E_deploy_max'] - E_deploy_acc)
    ub = min(ub, e_left / dt)

    # --- 4. SoC window -----------------------------------------------------
    #     Coulomb counting: dSoC = -I2 dt / Q_bat, with I2 ~= P2 / U.
    #     Discharge is limited so SoC_next >= SoC_min, charge so
    #     SoC_next <= SoC_max. Using U instead of the exact terminal voltage
    #     U2 is conservative for discharge (U2 < U -> real current is larger),
    #     so a small safety margin is applied.
    margin = 0.98
    I_dis_max = max(0.0, (SoC - params['SoC_min'])) * Q_bat / dt
    ub = min(ub, margin * I_dis_max * U)

    I_chg_max = max(0.0, (params['SoC_max'] - SoC)) * Q_bat / dt
    lb = max(lb, -margin * I_chg_max * U)

    # --- 5. availability of mechanical braking power ----------------------
    if P_gb < 0.0:
        # cannot absorb more braking power than the gearbox is shedding
        lb = max(lb, P_gb * eta_MGU)
    elif not allow_ice_charging:
        lb = max(lb, 0.0)

    # --- feasibility guard -------------------------------------------------
    if lb > ub:
        mid = 0.5 * (lb + ub)
        lb = ub = float(np.clip(mid, params['P_MGU_min'], params['P_MGU_max']))

    return float(lb), float(ub)


def project(action_raw, lb, ub, mode: str = "relative"):
    """
    Map a normalised agent action a in [-1, 1] to a physical P2 in [lb, ub].

    mode = "relative"
        a = -1 -> lb, a = +1 -> ub. The agent controls the *fraction of the
        currently available envelope* it wants to use. Pro: the whole action
        range is always meaningful, exploration never wastes samples on
        infeasible commands, and the policy transfers across states where the
        envelope has a very different width (e.g. before/after the deploy
        budget is exhausted). Con: the semantics of a given action value are
        state-dependent, which makes the learnt policy harder to inspect and
        makes the mapping non-stationary during an episode.

    mode = "absolute"
        a is scaled to the fixed regulatory envelope [P_MGU_min, P_MGU_max]
        and then clipped to [lb, ub]. Pro: fixed physical meaning, directly
        comparable to the DP/ECMS control trajectories. Con: large parts of
        the action range are inert whenever the envelope is narrow, which
        wastes exploration and flattens the policy gradient.

    Both are provided because the choice is a genuine design trade-off and the
    comparison is worth reporting.
    """
    a = float(np.clip(action_raw, -1.0, 1.0))
    if mode == "relative":
        return lb + 0.5 * (a + 1.0) * (ub - lb)
    elif mode == "absolute":
        raise_lo, raise_hi = -350e3, 350e3
        P2 = raise_lo + 0.5 * (a + 1.0) * (raise_hi - raise_lo)
        return float(np.clip(P2, lb, ub))
    raise ValueError(f"unknown action_mode: {mode}")


def inverse_project(P2, lb, ub, mode: str = "relative"):
    """Inverse of `project`, used to convert DP/ECMS trajectories into actions
    for behaviour cloning."""
    if mode == "relative":
        span = ub - lb
        if span <= 0.0:
            return 0.0
        return float(np.clip(2.0 * (P2 - lb) / span - 1.0, -1.0, 1.0))
    elif mode == "absolute":
        return float(np.clip(P2 / 350e3, -1.0, 1.0))
    raise ValueError(f"unknown action_mode: {mode}")
