import numpy as np
from .parameters import params

def mechanical_power(params,m_dot):
    eta_ICE = params['eta_ICE']
    P_ICE0 = params['P_ICE0'] #engine drag power
    LHV = params['LHV']


    Pf = LHV * m_dot #fuel_power
    m_dot_max = params['m_dot_max'] / 3600
    Pf_max = LHV * m_dot_max
    if Pf > Pf_max:
        Pf = Pf_max

    return eta_ICE*Pf - P_ICE0

    


def MGU_K(P_mech_desired, E_deploy_acc_k, E_recharge_acc_k, params, dt):
    """
    MGU-K component model: converts a desired mechanical power (from the
    power-split controller) into the corresponding electrical power P2 seen
    by the battery, enforcing the component's instantaneous power limits and
    the regulatory per-lap deploy energy budget.

    Sign convention (consistent with battery.py):
        P_mech_desired > 0 -> boost (motoring): MGU-K delivers mechanical
            power to the crankshaft, drawing electrical power from the battery.
        P_mech_desired < 0 -> regeneration (braking): MGU-K absorbs mechanical
            braking power and converts it to electrical power, charging the
            battery.

    Args:
        P_mech_desired (float): mechanical power requested from the MGU-K by
            the controller [W]. Positive = boost, negative = regen.
        E_deploy_acc_k (float): cumulative deploy (boost) energy used so far
            this lap [J]. State variable, must be passed in from the previous
            timestep.
        E_recharge_acc_k (float): cumulative regenerated energy so far this
            lap [J]. State variable, tracked for information/analysis; not
            currently capped (no K2ES-style limit defined in params).
        params (dict): must contain 'P_MGU_max' [W], 'P_MGU_min' [W]
            (negative), 'E_deploy_max' [J], 'eta_MGU' [-].
        dt (float): timestep duration [s].

    Returns:
        tuple:
            P2 (float): electrical power at the battery terminals [W].
                Positive = discharge, negative = charge.
            E_deploy_next (float): updated cumulative deploy energy [J],
                saturated at E_deploy_max.
            E_recharge_next (float): updated cumulative regenerated energy [J].
    """
    P_MGU_max = params['P_MGU_max']
    P_MGU_min = params['P_MGU_min']
    E_deploy_max = params['E_deploy_max']
    eta_MGU = params['eta_MGU']

    # Saturate the requested power to the MGU-K's instantaneous physical limits.
    P_mech_desired = np.clip(P_mech_desired, P_MGU_min, P_MGU_max)
    

    # If the lap's deploy budget is already exhausted, no further boosting
    #    is allowed for the rest of the lap (permanent saturation).

    if P_mech_desired > 0 and E_deploy_acc_k >= E_deploy_max:
        P_mech_desired = 0.0
    
    # Convert mechanical power to electrical power, direction-dependent
    if P_mech_desired > 0:
        P2 = P_mech_desired/eta_MGU
        dE_deploy = P2 * dt
        dE_recharge = 0
    elif P_mech_desired < 0:
        P2 = P_mech_desired*eta_MGU
        dE_deploy = 0
        dE_recharge = - P2 * dt #positive magnitude of energy recovered
    else:
        P2 = 0.0
        dE_deploy = 0.0
        dE_recharge = 0.0
    
    #Enforce the cumulative deploy budget: if this step would push the
    #accumulator past E_deploy_max, shrink this step's contribution so the
    #accumulator lands exactly on the cap, and reduce P2 consistently.
    E_deploy_next = E_deploy_acc_k + dE_deploy

    if E_deploy_next > E_deploy_max:
        dE_deploy = E_deploy_max - E_deploy_acc_k
        P2 = dE_deploy / dt if dt > 0 else 0.0
        E_deploy_next = E_deploy_max

    E_recharge_next = E_recharge_acc_k + dE_recharge

    return P2, E_deploy_next, E_recharge_next

def powertrain(P_gb_desired, u_split, E_deploy_acc_k, E_recharge_acc_k, params, dt):
    """
    Powertrain plant model: splits the requested wheel/gearbox power between
    the ICE and the MGU-K according to the controller's split decision,
    enforces each component's physical/regulatory limits, and reports any
    power shortfall if the combination cannot meet the request.

    Args:
        P_gb_desired (float): total mechanical power requested at the
            gearbox input [W], from vehicle_dynamics
        u_split (float): fraction of P_gb_desired requested from the MGU-K,
            in [0, 1].
        E_deploy_acc_k (float): cumulative MGU-K deploy (boost) energy used
            so far [J]. State variable, passed in from the previous timestep.
        E_recharge_acc_k (float): cumulative MGU-K regenerated energy so far
            [J]. State variable, passed in from the previous timestep.
        params (dict)
        dt (float): timestep duration [s].

    Returns:
        tuple:
            P_ICE_real (float): mechanical power actually delivered by the
                ICE [W], after applying the fuel flow limit.
            P2 (float): electrical power at the battery terminals [W]
                (output of the MGU-K model; positive = discharge).
            P_mech_MGU_K_real (float): mechanical power actually delivered
                (or absorbed) by the MGU-K [W], after applying its power
                limits and the deploy energy budget.
            shortfall (float): P_gb_desired - P_ICE_real - P_mech_MGU_K_real
                [W]. Zero if no constraint was active; nonzero means the
                powertrain cannot meet the requested power in this timestep
                given the controller's chosen split. Reported, not corrected,
                for the controller to handle.
            E_deploy_next (float): updated cumulative MGU-K deploy energy [J].
            E_recharge_next (float): updated cumulative MGU-K regenerated
                energy [J].
    """
    eta_ICE = params['eta_ICE']
    P_ICE0 = params['P_ICE0']
    LHV = params['LHV']
    m_dot_max_s = params['m_dot_max'] / 3600.0  # [kg/h] -> [kg/s]

    # Apply the controller's split to the desired total power.
    P_ICE_desired = (1 - u_split) * P_gb_desired
    P_mech_MGU_K_desired = u_split * P_gb_desired

    # Invert Willans to find the fuel mass flow needed for P_ICE_desired.
    m_dot = (P_ICE_desired + P_ICE0) / (eta_ICE * LHV)

    # Enforce the regulatory fuel flow limit, then recompute the actually
    # achievable ICE power from the (possibly clipped) mass flow, using
    # the forward Willans model for consistency with mechanical_power().
    m_dot = np.clip(m_dot, 0.0, m_dot_max_s)
    P_ICE_real = mechanical_power(params, m_dot)

    # Pass the MGU-K's desired power through its own component model,
    # which enforces P_MGU_min/max and the per-lap deploy energy budget.
    P2, E_deploy_next, E_recharge_next = MGU_K(P_mech_MGU_K_desired, E_deploy_acc_k, E_recharge_acc_k, params, dt)
    # The mechanical power actually delivered/absorbed by the MGU-K is
    # recovered from P2 using the same direction-dependent efficiency as
    # inside MGU_K (needed here only for the power balance/shortfall check).
    eta_MGU = params['eta_MGU']
    if P2 > 0:
        P_mech_MGU_K_real = P2 * eta_MGU
    elif P2 < 0:
        P_mech_MGU_K_real = P2 / eta_MGU
    else:
        P_mech_MGU_K_real = 0.0

    # Report, not correct, any shortfall between what was requested and
    #    what the powertrain can actually deliver given the chosen split.

    shortfall = P_gb_desired - P_ICE_real - P_mech_MGU_K_real

    return P_ICE_real, P2, P_mech_MGU_K_real, shortfall, E_deploy_next, E_recharge_next
