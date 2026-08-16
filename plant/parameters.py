# Technical regulations: F1 2026 power unit envelope

params = {
    # Vehicle
    'mv': 768,          # [kg] 2026 minimum weight, car + driver, without fuel
    'Cd': 0.7,          # [-] estimate, Z-mode (active aero not modeled)
    'Af': 1.5,          # [m^2] estimate, 1.9 m car width
    'Cr': 0.01,         # [-]

    # ICE (1.6L V6 turbo) - simplified Willans line
    'P_ICE_max': 400e3, # [W] 2026 ICE output
    'eta_ICE': 0.498,    # [-] peak efficiency
    'P_ICE0': 15e3,     # [W] engine drag power
    'LHV': 44e6,        # [J/kg]
    'm_dot_max': 68.18, # [kg/h] = 3000 MJ/h fuel energy flow limit / LHV
    'eta_gearbox': 0.97,

    # MGU-K
    'P_MGU_max': 350e3, # [W] 2026 ERS-K electrical DC power limit
    'P_MGU_min': -350e3,# [W] recovery, same ceiling as deploy
    'eta_MGU': 0.93,    # [-]

    # Energy Store
    'E_pack_capacity': 5.7e6,  # [J] sized so (SoC_max - SoC_min) gives ~4 MJ usable
    'SoC_max': 0.9,
    'SoC_min': 0.2,
    'R_int': 0.01,      # [Ohm]
    'V_oc_nom': 300,    # [V] nominal pack voltage

    # Per-lap energy limits
    'E_deploy_max': 9e6,  # [J] max ES->K energy per lap (HV DC bus)

    'rho_a': 1.225,
    'g': 9.81,

    # Thermal model - battery
    'c_p_cell': 900,
    'energy_density_Wh_per_kg': 220,
    'T_coolant_in': 50,
    'T_bat_safe_max': 60,
    'T_bat_derate_start': 45,
    'T_bat_init': 40,
}
