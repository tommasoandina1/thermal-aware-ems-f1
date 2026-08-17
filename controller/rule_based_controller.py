"""
Rule-based (SoC-scheduled) power-split controller.

Baseline heuristic used as the "no optimization" reference point against
which ECMS and DP are benchmarked in Compare_Controllers.ipynb. During
traction, the MGU-K power fraction `u_split` is scheduled linearly on SoC
between `u_min` and `u_max`, then scaled down as the per-lap deploy budget
(E_ES2K) is used up. During braking (P_gb_desired < 0), full regeneration
is requested (u_split = 1.0), subject to the plant's own physical limits.

Inputs:
    SoC_k           -- current state of charge [-]
    P_gb_desired    -- power requested at the gearbox [W]
    E_deploy_acc_k  -- energy deployed so far in the current lap [J]
    params          -- dict with 'SoC_max', 'SoC_min', 'E_deploy_max'

Returns:
    u_split -- MGU-K power fraction in [0, 1] (fraction of the traction
               request to be met electrically; ignored sign-wise during
               braking, where full regen is always requested)
"""

import numpy as np


def rule_based_split(SoC_k, P_gb_desired, E_deploy_acc_k, params):
    SoC_max = params['SoC_max']
    SoC_min = params['SoC_min']
    E_deploy_max = params['E_deploy_max']

    if P_gb_desired < 0:
        return 1.0

    u_min = 0.05
    u_max = 0.6
    SoC_clipped = np.clip(SoC_k, SoC_min, SoC_max)
    m = (u_max - u_min) / (SoC_max - SoC_min)
    q = u_min - m * SoC_min
    u_split = m * SoC_clipped + q

    deploy_margin = 1.0 - E_deploy_acc_k / E_deploy_max
    deploy_margin = np.clip(deploy_margin, 0.0, 1.0)
    u_split *= deploy_margin
    return np.clip(u_split, 0.0, 1.0)