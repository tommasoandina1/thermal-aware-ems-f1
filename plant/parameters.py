#Technical Regulations for LMDh Prototype

params = {
    # Veicolo
    'mv': 730,          # [kg] peso minimo LMDh con pilota
    'Cd': 0.7,          # [-] stima prototipo chiuso
    'Af': 1.2,          # [m^2]
    'Cr': 0.01,         # [-]
    'r_wheel': 0.33,    # [m]

    # ICE con turbo — Willans semplificato
    'P_ICE_max': 400e3, # [W] circa 680cv, limite LMDh
    'eta_ICE': 0.38,    # [-] picco rendimento
    'P_ICE0': 15e3,     # [W] drag power motore
    'LHV': 44e6,        # [J/kg] benzina
    'm_dot_max': 75,            # [kg/h] Fuel flow max
    'eta_gearbox': 0.97,

    # MGU-K
    'P_MGU_max': 350e3, # [W] limite regolamento F1
    'P_MGU_min': -200e3,# [W] recupero in frenata
    'eta_MGU': 0.93,    # [-]

    # Batteria
    'E_pack_capacity': 20e6,   # [J] capacità elettrochimica totale del pacco (~5.5kWh)
    'E_stint_max': 900e3,      # [J] vincolo regolamentare di scambio energetico per stint (se serve, tienilo distinto)
    'SoC_max': 0.9,
    'SoC_min': 0.2,
    'R_int': 0.01,      # [Ohm]
    'V_oc_nom': 300,    # [V] tensione nominale

    # Vincoli stint
    'E_deploy_max': 9e6,  # [J] energia deployment per giro

    'rho_a': 1.225,
    'g' : 9.81,
}
