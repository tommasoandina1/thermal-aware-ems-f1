"""
Baseline controllers expressed as policies over the RL environment.

The point of this module is methodological rather than algorithmic: the
rule-based, ECMS and DP controllers already exist in the notebooks, but each
runs its own simulation loop. Any comparison built on three different loops is
vulnerable to the objection that the difference comes from the harness and not
from the strategy. Re-running every baseline *inside EMSEnv* removes that
objection: identical plant, identical safety layer, identical accounting of
fuel, shortfall and terminal SoC.

Expect small numerical differences with respect to the notebook figures. They
have one identified cause: the safety layer bounds the command before the plant
sees it, whereas the notebooks let battery_step() clip the battery power
internally. The safety-layer version is the more consistent of the two, because
after clipping the requested and the delivered power no longer agree and the
shortfall reported by powertrain() stops being the single source of truth.
"""

from __future__ import annotations
import numpy as np

from plant.parameters import params as DEFAULT_PARAMS
from plant.battery import Uoc
from controller.rule_based_controller import rule_based_split

from .safety import inverse_project


def _u_split_to_P2(u, P_gb, params):
    """Convert a mechanical split fraction into an electrical command."""
    eta = params["eta_MGU"]
    P_mech = u * P_gb if P_gb > 0 else u * P_gb   # u=1 during braking -> full regen
    if P_mech > 0:
        return P_mech / eta
    elif P_mech < 0:
        return P_mech * eta
    return 0.0


class RuleBasedPolicy:
    """SoC-scheduled heuristic, wrapped as an env policy."""

    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS

    def __call__(self, env, obs):
        s = env.state
        P_gb = float(env.prof["P_gb"][s["k"]])
        u = rule_based_split(s["SoC"], P_gb, s["E_deploy"], self.params)
        P2 = _u_split_to_P2(u, P_gb, self.params)
        lb, ub = env._bounds()
        return np.array([inverse_project(np.clip(P2, lb, ub), lb, ub,
                                        env.cfg.action_mode)])


class ECMSPolicy:
    """
    Equivalent Consumption Minimisation Strategy with SoC feedback on the
    equivalence factor.

        s(t) = s_0 - Kp * (SoC(t) - SoC_ref)

    The instantaneous problem is solved by direct search over the admissible
    electrical power interval, which is both simple and robust: the cost is not
    smooth (fuel-flow saturation, deploy budget, thermal ceiling) so a
    closed-form stationarity condition would be misleading at exactly the
    operating points that matter.

    Defaults follow the corrected calibration: s_0 = 1 / eta_ICE (the physically
    meaningful equivalence between electrical and fuel energy) and a small
    proportional gain. A large gain drives the controller into bang-bang
    behaviour, which is a controller pathology, not an optimisation result.
    """

    def __init__(self, s0=None, Kp=5.0, soc_ref=None, n_grid=41, params=None,
                 rate_penalty=0.0):
        self.params = params or DEFAULT_PARAMS
        self.s0 = s0 if s0 is not None else 1.0 / self.params["eta_ICE"]
        self.Kp = Kp
        self.soc_ref = soc_ref
        self.n_grid = n_grid
        self.rate_penalty = rate_penalty

    def __call__(self, env, obs):
        p = self.params
        st = env.state
        P_gb = float(env.prof["P_gb"][st["k"]])
        soc_ref = self.soc_ref if self.soc_ref is not None else env._soc_target
        s_eq = self.s0 - self.Kp * (st["SoC"] - soc_ref)

        lb, ub = env._bounds()
        if ub - lb < 1.0:
            grid = np.array([0.5 * (lb + ub)])
        else:
            grid = np.linspace(lb, ub, self.n_grid)

        eta_MGU, eta_ICE, LHV = p["eta_MGU"], p["eta_ICE"], p["LHV"]
        P_ICE0, m_dot_max = p["P_ICE0"], p["m_dot_max"] / 3600.0

        P_mech = np.where(grid > 0, grid * eta_MGU, grid / eta_MGU)
        P_ICE_des = P_gb - P_mech
        m_dot = np.clip((P_ICE_des + P_ICE0) / (eta_ICE * LHV), 0.0, m_dot_max)
        P_ICE_real = eta_ICE * LHV * m_dot - P_ICE0
        shortfall = np.maximum(0.0, P_gb - P_ICE_real - P_mech)

        # Hamiltonian: fuel power + s * electrical power, plus an explicit
        # shortfall price so the search does not "solve" the problem by simply
        # leaving demand unmet.
        H = LHV * m_dot + s_eq * grid + 50.0 * shortfall
        if self.rate_penalty > 0.0:
            H = H + self.rate_penalty * np.abs(grid - st["P2_prev"])

        P2 = float(grid[int(np.argmin(H))])
        return np.array([inverse_project(P2, lb, ub, env.cfg.action_mode)])


class ReplayPolicy:
    """
    Replays a precomputed control trajectory (e.g. the DP optimal policy
    exported from notebook 03/04) inside the environment.

    Parameters
    ----------
    P2_traj : array of length N, electrical power command per step [W].
    """

    def __init__(self, P2_traj):
        self.P2 = np.asarray(P2_traj, float)

    def __call__(self, env, obs):
        k = min(env.state["k"], len(self.P2) - 1)
        lb, ub = env._bounds()
        return np.array([inverse_project(float(np.clip(self.P2[k], lb, ub)),
                                         lb, ub, env.cfg.action_mode)])


class ConstantPolicy:
    def __init__(self, a=0.0):
        self.a = a

    def __call__(self, env, obs):
        return np.array([self.a])
