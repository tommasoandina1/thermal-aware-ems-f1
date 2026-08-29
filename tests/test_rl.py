"""
Tests for the RL layer.

The purpose is not coverage for its own sake. Each test pins down one property
that, if it broke silently, would produce plausible-looking but meaningless
results - the failure mode that is hardest to catch by looking at a learning
curve.

Run with:  python -m pytest tests/test_rl.py -q
"""

import numpy as np
import pytest

from plant.parameters import params
from plant.battery import coef_cubic, thermal_derating_factor
from rl.config import EnvConfig
from rl.env import EMSEnv, load_profile
from rl.safety import feasible_P2_bounds, project, inverse_project
from rl.baselines import RuleBasedPolicy, ECMSPolicy, ConstantPolicy
from rl.evaluate import rollout
from rl.buffer import ReplayBuffer


@pytest.fixture(scope="module")
def prof1():
    return load_profile(EnvConfig(profile="single_lap"))


@pytest.fixture
def env(prof1):
    return EMSEnv(EnvConfig(profile="single_lap"), profile=prof1)


# ---------------------------------------------------------------------------
# safety layer
# ---------------------------------------------------------------------------

def test_bounds_never_inverted():
    """lb > ub fed to a sampler is silently meaningless, so it must be
    impossible by construction - including at the corners of the state space."""
    for SoC in (params["SoC_min"], 0.5, params["SoC_max"]):
        for T in (25.0, 45.0, 55.0, 70.0):
            for E in (0.0, params["E_deploy_max"], 2 * params["E_deploy_max"]):
                for P_gb in (-6e5, 0.0, 6e5):
                    lb, ub = feasible_P2_bounds(SoC, T, E, P_gb, params, 0.1,
                                                coef_cubic)
                    assert lb <= ub + 1e-9, (SoC, T, E, P_gb, lb, ub)


def test_bounds_respect_regulatory_envelope():
    lb, ub = feasible_P2_bounds(0.6, 40.0, 0.0, 5e5, params, 0.1, coef_cubic)
    assert ub <= params["P_MGU_max"] + 1e-6
    assert lb >= params["P_MGU_min"] - 1e-6


def test_deploy_budget_closes_the_upper_bound():
    _, ub = feasible_P2_bounds(0.6, 40.0, params["E_deploy_max"], 5e5,
                               params, 0.1, coef_cubic)
    assert ub == pytest.approx(0.0, abs=1e-6)


def test_thermal_derating_shrinks_the_upper_bound():
    _, ub_cold = feasible_P2_bounds(0.6, 40.0, 0.0, 5e5, params, 0.1, coef_cubic)
    _, ub_hot = feasible_P2_bounds(0.6, 55.0, 0.0, 5e5, params, 0.1, coef_cubic)
    assert thermal_derating_factor(55.0, params) < 1.0
    assert ub_hot < ub_cold


def test_regen_cannot_exceed_available_braking_power():
    P_gb = -3e5
    lb, _ = feasible_P2_bounds(0.5, 40.0, 0.0, P_gb, params, 0.1, coef_cubic)
    assert lb >= P_gb * params["eta_MGU"] - 1e-6


def test_projection_roundtrip():
    lb, ub = -2e5, 3e5
    for a in np.linspace(-1, 1, 11):
        P2 = project(a, lb, ub, "relative")
        assert lb - 1e-6 <= P2 <= ub + 1e-6
        assert inverse_project(P2, lb, ub, "relative") == pytest.approx(a, abs=1e-6)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def test_observation_shape_and_finiteness(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    for _ in range(50):
        obs, r, term, trunc, _ = env.step([0.1])
        assert np.all(np.isfinite(obs)) and np.isfinite(r)


@pytest.mark.parametrize("a", [-1.0, -0.3, 0.0, 0.7, 1.0])
def test_hard_constraints_hold_for_any_action(env, a):
    """With the safety layer on, no constant action may violate the SoC window,
    the thermal safety limit or the per-lap deploy budget."""
    env.reset(seed=0)
    while True:
        _, _, term, trunc, info = env.step([a])
        assert params["SoC_min"] - 1e-9 <= info["SoC"] <= params["SoC_max"] + 1e-9
        assert info["T_bat"] <= params["T_bat_safe_max"] + 5.0
        assert info["E_deploy"] <= params["E_deploy_max"] + 1e-6
        if term or trunc:
            break


def test_no_silent_clipping_by_the_plant(env):
    """The safety layer must pre-empt battery_step()'s internal power clip,
    otherwise requested and delivered power diverge and the shortfall reported
    by powertrain() stops being the single source of truth."""
    env.reset(seed=0)
    while True:
        _, _, term, trunc, info = env.step([1.0])
        assert not info["silent_clip"]
        if term or trunc:
            break


def test_deploy_budget_resets_at_lap_boundary(prof1):
    cfg = EnvConfig(profile="multi_lap", n_laps=2)
    env = EMSEnv(cfg)
    env.reset(seed=0)
    saw_reset = False
    prev = 0.0
    while True:
        _, _, term, trunc, info = env.step([1.0])
        if info["lap_ended"]:
            assert info["E_deploy"] == pytest.approx(0.0)
            saw_reset = True
        prev = info["E_deploy"]
        if term or trunc:
            break
    assert saw_reset


def test_reward_is_negative_gram_equivalent_cost(env):
    """One step of pure fuel burn with no shortfall and no rate change must
    return exactly -(grams burnt) * reward_scale."""
    env.cfg.w_rate = 0.0
    env.reset(seed=0)
    _, r, _, _, info = env.step([0.0])
    expected = -(info["fuel_g"] + env.cfg.w_shortfall
                 * max(0.0, info["shortfall_W"]) * env.dt / 1e6) * env.cfg.reward_scale
    assert r == pytest.approx(expected, rel=1e-9)


def test_shortfall_penalty_is_continuous_not_binary(env):
    """A larger shortfall must be strictly worse. A binary penalty makes the
    optimiser indifferent between a 0.1 MJ and a 12 MJ shortfall."""
    from rl.env import EMSEnv as E
    r = []
    for a in (1.0, -1.0):
        e = E(env.cfg, profile=env.prof)
        e.reset(seed=0)
        tot_short, tot_r = 0.0, 0.0
        for _ in range(200):
            _, rr, _, _, info = e.step([a])
            tot_short += max(0.0, info["shortfall_W"]) * e.dt / 1e6
            tot_r += rr
        r.append((tot_short, tot_r))
    (s_hi, r_hi), (s_lo, r_lo) = sorted(r, key=lambda x: -x[0])
    assert s_hi > s_lo
    assert r_hi != r_lo


def test_soc_equalised_fuel_is_the_reported_metric(env):
    env.reset(seed=0)
    while True:
        _, _, term, trunc, info = env.step([1.0])
        if term or trunc:
            break
    s = info["episode"]
    g_per_soc = params["E_pack_capacity"] / (params["eta_ICE"] * params["LHV"]) * 1000
    expected = s["fuel_g"] + (s["SoC_target"] - s["SoC_final"]) * g_per_soc
    assert s["fuel_eq_g"] == pytest.approx(expected, rel=1e-9)
    # discharging the pack must make the equalised figure worse than raw fuel
    assert s["SoC_final"] < s["SoC_target"] and s["fuel_eq_g"] > s["fuel_g"]


def test_determinism_given_a_seed(prof1):
    cfg = EnvConfig(profile="single_lap", soc_init_jitter=0.05, tbat_init_jitter=5.0)
    outs = []
    for _ in range(2):
        e = EMSEnv(cfg, profile=prof1)
        outs.append(rollout(e, ConstantPolicy(0.3), seed=7)["fuel_eq_g"])
    assert outs[0] == pytest.approx(outs[1], rel=1e-12)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def test_rule_based_reproduces_the_repository_figures(prof1):
    """Anchor test. The rule-based controller run inside EMSEnv must land on
    the canonical single-lap figures (773.5 g, 2.02 MJ, SoC_f 0.5417). Tight
    but not exact: the safety layer bounds the command before the plant sees
    it, where the notebook lets the battery model clip internally."""
    env = EMSEnv(EnvConfig(profile="single_lap"), profile=prof1)
    out = rollout(env, RuleBasedPolicy(), seed=0)
    assert out["fuel_g"] == pytest.approx(773.5, rel=0.02)
    assert out["shortfall_MJ"] == pytest.approx(2.02, rel=0.05)
    assert out["SoC_final"] == pytest.approx(0.5417, abs=0.02)


def test_ecms_beats_rule_based_on_unmet_demand(prof1):
    mk = lambda: EMSEnv(EnvConfig(profile="single_lap"), profile=prof1)
    rb = rollout(mk(), RuleBasedPolicy(), seed=0)
    ec = rollout(mk(), ECMSPolicy(), seed=0)
    assert ec["shortfall_MJ"] < rb["shortfall_MJ"]


def test_high_gain_ecms_chatters(prof1):
    """Chattering in ECMS is structural - it comes from the absence of a rate
    term, not from budget exhaustion. A large proportional gain must produce
    markedly more command switching, and the rate penalty must suppress it."""
    mk = lambda: EMSEnv(EnvConfig(profile="single_lap"), profile=prof1)
    def switches(policy):
        out = rollout(mk(), policy, seed=0, record=True)
        P2 = out["traj"]["P2"]
        return int(np.sum(np.abs(np.diff(np.sign(np.diff(P2)))) > 0))
    hi = switches(ECMSPolicy(s0=1.0, Kp=170.0))
    lo = switches(ECMSPolicy(s0=1.0 / params["eta_ICE"], Kp=5.0))
    assert hi > lo


# ---------------------------------------------------------------------------
# buffer
# ---------------------------------------------------------------------------

def test_buffer_roundtrip_and_priorities():
    b = ReplayBuffer(4, 1, capacity=100, prioritized=True)
    for i in range(150):
        b.add(np.full(4, i, np.float32), np.array([0.1]), -float(i),
              np.full(4, i + 1, np.float32), 0.0, cost=0.01 * i)
    assert len(b) == 100
    batch = b.sample(16)
    assert batch["obs"].shape == (16, 4) and batch["weights"].shape == (16, 1)
    b.update_priorities(batch["idx"], np.ones(16))
    assert np.all(b.prio[batch["idx"]] > 0)


# ---------------------------------------------------------------------------
# agent (imported lazily: torch is only needed for these)
# ---------------------------------------------------------------------------

def test_sac_update_runs_and_changes_parameters(prof1):
    torch = pytest.importorskip("torch")
    from rl.sac import SAC
    from rl.config import SACConfig

    env = EMSEnv(EnvConfig(profile="single_lap"), profile=prof1)
    obs, _ = env.reset(seed=0)
    ag = SAC(env.observation_space.shape[0], 1, SACConfig(use_lagrangian=True))
    buf = ReplayBuffer(env.observation_space.shape[0], 1, 5000)
    for _ in range(400):
        a = ag.act(obs)
        nobs, r, t1, t2, info = env.step(a)
        buf.add(obs, a, r, nobs, float(t1), cost=info["cost"])
        obs = nobs
        if t1 or t2:
            obs, _ = env.reset(seed=1)

    before = [p.detach().clone() for p in ag.actor.parameters()]
    for _ in range(10):
        ag.update(buf.sample(64))
    after = list(ag.actor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert np.isfinite(float(ag.alpha))
    assert ag.update_lambda(0.5)["lam"] > 0
