import numpy as np


def  rule_based_split(SoC_k, P_gb_desired,E_deploy_acc_k, params):
    SoC_max=params['SoC_max']
    SoC_min = params['SoC_min']
    E_deploy_max = params['E_deploy_max']

    if P_gb_desired<0:
        return 1.0
    u_min = 0.05
    u_max = 0.3
    SoC_clipped = np.clip(SoC_k, SoC_min, SoC_max)
    m = (u_max-u_min)/(SoC_max-SoC_min)
    q = u_min - m * SoC_min
    u_split = m * SoC_clipped + q

    deploy_margin = 1.0 - E_deploy_acc_k / E_deploy_max
    deploy_margin = np.clip(deploy_margin, 0.0, 1.0)
    u_split *= deploy_margin
    return np.clip(u_split, 0.0, 1.0)
