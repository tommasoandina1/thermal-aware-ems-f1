"""
battery.py

Quasistatic equivalent circuit battery model for an F1 car.

The battery pack is modeled with a single open-circuit voltage source Uoc(SoC)
in series with a constant internal resistance R_int (no dynamic RC branches,
consistent with a quasistatic simulation rather than a real-time BMS estimator).

Workflow per timestep:
    1. Read current SoC.
    2. Read requested battery power P2 (from the power-split controller).
    3. Solve the equivalent-circuit quadratic equation for terminal voltage U2.
    4. Derive terminal current I2 = P2 / U2.
    5. Integrate SoC forward using Coulomb counting.

Sign convention: P2 > 0 means the battery is discharging (delivering power to
the MGU-K), P2 < 0 means the battery is charging (regenerative braking).
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import params

# ---------------------------------------------------------------------------
# OCV(SoC) CURVE CALIBRATION
# ---------------------------------------------------------------------------
# Reference cell data: Samsung INR21700-48X discharge OCV curve (Rukavina et
# al., 2023), read approximately from the published SoC-OCV plot. Used only
# as a physically plausible reference shape.

SoC = np.linspace(0., 1, num=11)
Uoc_cell = np.array([2.5, 3.3, 3.45, 3.55, 3.62, 3.68, 3.72, 3.78, 3.85, 3.95, 4.2])


# Linear rescaling of the single-cell OCV curve to the target pack voltage
# window, anchored at the operating SoC bounds (SoC_min=0.2 -> 270 V, SoC=0.8 -> 330 V)

x1, y1 = 3.45, 270
x2, y2 = 3.85, 330

m = (y2 - y1) / (x2 - x1)
q = y1 - m * x1
Uoc_pack = m * Uoc_cell + q

# Fit only over the operating SoC window (0.2-0.9) to avoid wasting fit
# accuracy on SoC extremes the pack never reaches.

coef_cubic = np.polyfit(SoC[2:10],Uoc_pack[2:10],3)
coef_quad = np.polyfit(SoC[2:10],Uoc_pack[2:10],2)


def Uoc(x, coef):
    """
    Evaluate the fitted open-circuit voltage of the pack at a given SoC.

    Args:
        SoC_value (float): state of charge, in [0, 1].
        coef (array-like): polynomial coefficients (highest degree first),
            as returned by np.polyfit. Works for any polynomial degree.

    Returns:
        float: pack open-circuit voltage Uoc(SoC) [V].
    """
    poly = np.poly1d(coef)
    return poly(x)

def battery_step(SoC_k, P2_k, coef, params,dt):
    """
    Battery state by one quasistatic timestep.

    Solves the equivalent-circuit quadratic equation
        U2^2 - Uoc*U2 + P2*R_int = 0
    for the terminal voltage U2, then derives current and updates SoC via
    Coulomb counting.

    Args:
        SoC_k (float): state of charge at the current timestep, in [0, 1].
        P2_k (float): requested battery terminal power [W].
            Positive = discharge, negative = charge (regenerative braking).
        coef (array-like): Uoc(SoC) polynomial fit coefficients.
        params (dict): must contain 'R_int' [Ohm] and 'Q_bat' [C].
        dt (float): timestep duration [s].

    Returns:
        tuple:
            SoC_next (float): state of charge at the next timestep, clipped
                to [params['SoC_min'], params['SoC_max']].
            U2_k (float): terminal voltage at this timestep [V].
            I2_k (float): terminal current at this timestep [A].
    """
    Uoc_k = Uoc(SoC_k, coef)
    R_int = params['R_int']
    Q_bat = params['E_pack_capacity'] /params['V_oc_nom'] 
    # Maximum physically deliverable discharge power at this SoC/R_int
    
    P2_max = Uoc_k**2 /(4*R_int)
    if P2_k > P2_max:
        P2_k = P2_max

    U2_k = (Uoc_k + np.sqrt(Uoc_k**2 -4 * P2_k * R_int))/2
    I2_k = P2_k / U2_k
    dSoC = -I2_k * dt / Q_bat

    SoC_next = SoC_k + dSoC

    SoC_next = np.clip(SoC_next, params['SoC_min'], params['SoC_max'])
    return SoC_next, U2_k, I2_k


if __name__ == "__main__":
    # Calibration plot: single-cell curve, rescaled pack curve, and both
    # polynomial fits over the operating SoC window. Run this file directly
    # to regenerate the figure; not executed on import.
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(SoC, Uoc_cell)
    ax1.set_ylabel("Uoc_cell [V]")

    ax2.plot(SoC, Uoc_pack, label='estimation')
    ax2.plot(SoC[2:10], np.poly1d(coef_cubic)(SoC[2:10]), label='cubic fit')
    ax2.plot(SoC[2:10], np.poly1d(coef_quad)(SoC[2:10]), label='quadratic fit')
    ax2.set_ylabel("Uoc_pack [V]")
    ax2.set_xlabel("SoC [-]")
    ax2.legend()

    plt.savefig('soc.png')
    plt.show()
