"""
Tests for the thermal derating semantics of the plant.

These pin down the change that moved the derating from the battery power
ceiling to the MGU-K deploy envelope. Each one blocks a specific way the old
behaviour could creep back in.

Run with:  python -m pytest tests/test_plant_thermal.py -q
"""

import warnings

import numpy as np
import pytest

from plant.parameters import params
from plant.battery import (Uoc, coef_cubic, battery_step, thermal_model,
                           thermal_derating_factor)
from plant.powertrain import MGU_K, powertrain

DT = 0.1
T_START = params["T_bat_derate_start"]
T_MAX = params["T_bat_safe_max"]


# ---------------------------------------------------------------------------
# where the derating lives now
# ---------------------------------------------------------------------------

def test_derating_caps_the_mguk_deploy_envelope():
    """The whole point of the change: at elevated temperature the MGU-K may
    not deploy its full 350 kW."""
    for T, expected in [(T_START, 1.0), (50.0, 2 / 3), (55.0, 1 / 3), (T_MAX, 0.0)]:
        P2, _, _ = MGU_K(1e6, 0.0, 0.0, params, DT, Tbat_k=T)
        assert P2 == pytest.approx(params["P_MGU_max"] * expected, rel=1e-9)


def test_derating_is_monotone_in_temperature():
    Ts = np.linspace(T_START - 5, T_MAX + 5, 25)
    caps = [MGU_K(1e6, 0.0, 0.0, params, DT, Tbat_k=T)[0] for T in Ts]
    assert np.all(np.diff(caps) <= 1e-6)


def test_regeneration_is_not_derated():
    """The derating models the deploy-side thermal limit. Recovering energy
    while hot must stay possible - forbidding it would be a modelling claim
    nothing in the project supports."""
    cold, _, _ = MGU_K(-3e5, 0.0, 0.0, params, DT, Tbat_k=30.0)
    hot, _, _ = MGU_K(-3e5, 0.0, 0.0, params, DT, Tbat_k=58.0)
    assert cold == pytest.approx(hot)
    assert hot < 0.0


def test_no_temperature_means_no_thermal_limiting():
    """Backward compatibility: notebooks 01-03 call these functions without a
    temperature and must keep their published numbers."""
    for T_hot in (55.0, 59.0):
        without, _, _ = MGU_K(1e6, 0.0, 0.0, params, DT)
        with_hot, _, _ = MGU_K(1e6, 0.0, 0.0, params, DT, Tbat_k=T_hot)
        assert without == pytest.approx(params["P_MGU_max"])
        assert with_hot < without


# ---------------------------------------------------------------------------
# where it no longer lives
# ---------------------------------------------------------------------------

def test_battery_step_no_longer_thermally_clips():
    """battery_step must return the same result hot and cold for any command
    the component limits actually allow."""
    P2 = params["P_MGU_max"]
    cold = battery_step(0.6, P2, coef_cubic, 30.0, params, DT)
    hot = battery_step(0.6, P2, coef_cubic, 59.0, params, DT)
    assert cold[0] == pytest.approx(hot[0])
    assert cold[1] == pytest.approx(hot[1])
    assert cold[2] == pytest.approx(hot[2])


def test_solvability_guard_warns_instead_of_clipping_silently():
    """Exceeding Uoc^2/(4 R_int) means the caller skipped the component
    limits. It has to be loud, because a silent clip breaks the power balance:
    requested and delivered power stop agreeing and the shortfall reported by
    powertrain() is no longer the single source of truth."""
    U = float(Uoc(0.6, coef_cubic))
    P2_solvable = U ** 2 / (4 * params["R_int"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SoC_next, U2, I2 = battery_step(0.6, 1.5 * P2_solvable, coef_cubic,
                                        40.0, params, DT)
        assert len(w) == 1 and issubclass(w[0].category, RuntimeWarning)
    assert np.isreal(U2) and np.isfinite(U2)


def test_no_warning_in_the_normal_operating_range():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for SoC in np.linspace(params["SoC_min"], params["SoC_max"], 15):
            for P2 in np.linspace(params["P_MGU_min"], params["P_MGU_max"], 15):
                battery_step(SoC, P2, coef_cubic, 55.0, params, DT)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# the reason for the change: the constraint must reach the power balance
# ---------------------------------------------------------------------------

def test_thermal_limit_shows_up_as_shortfall():
    """The forward pass must now report the power that the thermal limit
    prevented from being delivered. Before the change this power simply
    vanished, so the plant was not thermally aware even though the DP bound
    was."""
    P_gb = 7e5
    control = 3e5          # electrical command, within the cold envelope
    cold = powertrain(P_gb, control, 0.0, 0.0, params, DT,
                      control_mode="P2", Tbat_k=30.0)
    hot = powertrain(P_gb, control, 0.0, 0.0, params, DT,
                     control_mode="P2", Tbat_k=57.0)
    assert hot[1] < cold[1]                 # delivered electrical power capped
    assert hot[3] > cold[3] + 1e3           # shortfall grows accordingly


def test_power_balance_closes_when_hot():
    """shortfall == P_gb - P_ICE - P_mech_MGU, exactly, at any temperature."""
    for T in (30.0, 48.0, 55.0, 59.5):
        P_ICE, P2, P_mech, sf, *_ = powertrain(6e5, 2.5e5, 0.0, 0.0, params, DT,
                                               control_mode="P2", Tbat_k=T)
        assert sf == pytest.approx(6e5 - P_ICE - P_mech, rel=1e-9, abs=1e-6)


def test_deploy_budget_and_derating_compose():
    """Both limits active at once must give the tighter of the two, not their
    product and not the last one applied."""
    E_used = params["E_deploy_max"] - 1e4      # 10 kJ left -> 100 kW for 0.1 s
    P2, E_next, _ = MGU_K(1e6, E_used, 0.0, params, DT, Tbat_k=50.0)
    derated = params["P_MGU_max"] * thermal_derating_factor(50.0, params)
    assert P2 == pytest.approx(min(derated, 1e4 / DT))
    assert E_next <= params["E_deploy_max"] + 1e-9


# ---------------------------------------------------------------------------
# closed loop
# ---------------------------------------------------------------------------

def test_thermal_runaway_is_self_limiting():
    """Held at full deploy request with degraded cooling, the temperature must
    approach the safety limit and stop, because the derating drives the
    admissible power to zero. If this diverges, the derating is not reaching
    the plant."""
    SoC, T, E_dep = 0.9, 40.0, 0.0
    for _ in range(4000):
        P_ICE, P2, P_mech, sf, E_dep, E_rec, m_dot = powertrain(
            7e5, params["P_MGU_max"], E_dep, 0.0, params, DT,
            control_mode="P2", Tbat_k=T)
        SoC, U2, I2 = battery_step(SoC, P2, coef_cubic, T, params, DT)
        T = thermal_model(I2, T, params, DT, cooling_factor=0.3)
        if E_dep >= params["E_deploy_max"]:
            E_dep = 0.0                       # lap boundary
    assert T < params["T_bat_safe_max"] + 1e-6
    assert np.isfinite(T)
