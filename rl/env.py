"""
Gymnasium environment wrapping the existing quasistatic powertrain plant.

Design notes
------------

MARKOV STATE. The observation contains every quantity whose value is needed to
predict the future evolution of the plant:

    SoC          - battery state of charge
    T_bat        - battery temperature (drives the derating factor)
    E_deploy_acc - per-lap deploy energy already consumed (cumulative, reset at
                   each lap boundary; without it the transition is not Markov)
    P_gb, v, a   - current operating point
    lap phase    - normalised position within the lap
    laps left    - the problem is finite-horizon and non-stationary: the optimal
                   policy depends on how many laps remain (this is exactly why
                   the baseline multi-lap DP drains the pack in the first two
                   laps)
    preview      - k samples of future gearbox power demand

The last block is a deliberate modelling choice. The DP benchmark is
clairvoyant; an agent with no preview would be solving a strictly harder
(partially observable) problem and the RL-vs-DP gap would conflate two distinct
sources of degradation (function approximation and partial observability). A
short preview window is also realistic: in a race the track map is known and the
speed trace is highly repeatable lap to lap. Setting preview_horizon = 0
recovers the POMDP formulation and is the natural ablation to report.

REWARD. Expressed in gram-equivalents of fuel, so every weight is physically
anchored:

    fuel term      m_dot * dt * 1000                        [g]
    shortfall      w_shortfall * (shortfall * dt / 1e6)     [g per MJ unmet]
    rate           w_rate * |dP2| / 1e6                     [g per MW step]
    SoC (lap end)  w_soc_lin * h + w_soc_quad * h^2, h = max(0, SoC_tgt - SoC)

The natural equivalence for the shortfall term is 1 MJ of unmet gearbox energy
= 1e6 / (eta_ICE * LHV) = 45.6 g of fuel. The default w_shortfall = 500 g/MJ is
therefore roughly 10x the natural price: unmet demand is treated as much worse
than the fuel it would have cost, which reflects the fact that a shortfall is a
lap-time loss and not a fuel expense. The same reasoning fixes w_soc_lin: one
unit of SoC is E_pack / (eta_ICE * LHV) * 1000 = 260 g of fuel.

Critically the shortfall penalty is CONTINUOUS in the shortfall magnitude, not
a binary large constant. A binary penalty makes the optimiser indifferent
between a 0.1 MJ and a 12 MJ shortfall and provides no useful gradient - a
failure mode already observed in the DP formulation and much more damaging in
RL, where the gradient is the only learning signal.

COST SIGNAL. info["cost"] carries the per-step shortfall energy in MJ, so the
same environment can be used either in the scalarised form above or in the CMDP
form (minimise fuel subject to E[sum cost] <= budget) via the Lagrangian option
in rl/sac.py. In the CMDP mode the shortfall term should be removed from the
reward (set w_shortfall = 0) so that the constraint is not double-counted.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from plant.parameters import params as DEFAULT_PARAMS
from plant.vehicle_dynamics import longitudinal_dynamics
from plant.battery import battery_step, thermal_model, thermal_derating_factor, coef_cubic
from plant.powertrain import powertrain

from .config import EnvConfig
from .safety import feasible_P2_bounds, project


# ---------------------------------------------------------------------------
# profile loading
# ---------------------------------------------------------------------------

def load_profile(cfg: EnvConfig, params=None):
    """
    Load the telemetry profile and precompute the gearbox power demand.

    Returns a dict with t, v, a, lap, P_gb, all 1-D arrays of equal length.
    P_gb is recomputed here rather than read from the .npy produced by
    scripts/simulation.py, so that the RL layer has no ordering dependency on
    that script.
    """
    params = params or DEFAULT_PARAMS
    root = cfg.data_root.rstrip("/")

    if cfg.profile == "single_lap":
        d = np.load(f"{root}/data/qualifying_Canada/Canada_qualifying.npy")
        t, v, a = d[0], d[1], d[2]
        lap = np.ones_like(t)
    elif cfg.profile == "multi_lap":
        d = np.load(f"{root}/data/multi_lap_Canada/Canada_5laps.npy")
        t, lap, v, a = d[0], d[1], d[2], d[3]
    else:
        raise ValueError(f"unknown profile: {cfg.profile}")

    if cfg.n_laps is not None:
        laps_unique = np.unique(lap)
        keep = np.isin(lap, laps_unique[: cfg.n_laps])
        t, lap, v, a = t[keep], lap[keep], v[keep], a[keep]

    P_gb = np.empty_like(v, dtype=float)
    for k in range(len(v)):
        _, _, P_gb[k] = longitudinal_dynamics(v[k], a[k], params)

    return dict(t=np.asarray(t, float), v=np.asarray(v, float),
                a=np.asarray(a, float), lap=np.asarray(lap, float),
                P_gb=P_gb)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

class EMSEnv(gym.Env):
    """Energy-management environment for the hybrid power unit."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, params=None, profile=None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.params = dict(params or DEFAULT_PARAMS)
        self.coef = coef_cubic
        self.prof = profile if profile is not None else load_profile(self.cfg, self.params)
        self.N = len(self.prof["t"])
        self.dt = float(self.cfg.dt)

        # normalisation constants: fixed and analytic, no running statistics,
        # so that a checkpoint is reproducible without carrying extra state
        self._P_SCALE = 8e5          # [W] covers the observed |P_gb| range
        self._V_SCALE = 100.0        # [m/s]
        self._A_SCALE = 30.0         # [m/s^2]
        # temperature window: covers cold start through the hard safety limit
        self._T_LO = 25.0
        self._T_HI = float(self.params['T_bat_safe_max']) + 5.0

        n_obs = 8 + self.cfg.preview_horizon
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        self._rng = np.random.default_rng()
        self.state = None

    # -- helpers -----------------------------------------------------------

    def _preview(self, k):
        h, s = self.cfg.preview_horizon, self.cfg.preview_stride
        if h == 0:
            return np.zeros(0, dtype=np.float64)
        idx = np.clip(k + s * np.arange(1, h + 1), 0, self.N - 1)
        return self.prof["P_gb"][idx] / self._P_SCALE

    def _obs(self):
        s = self.state
        k = s["k"]
        p = self.params
        lap_len = self._lap_len[s["lap_idx"]]
        lap_pos = (k - self._lap_start[s["lap_idx"]]) / max(1, lap_len)
        laps_left = (self._n_laps - s["lap_idx"] - 1) / max(1, self._n_laps)

        soc_n = (s["SoC"] - p["SoC_min"]) / (p["SoC_max"] - p["SoC_min"]) * 2 - 1
        tb_n = (s["T_bat"] - self._T_LO) / (self._T_HI - self._T_LO) * 2 - 1
        dep_n = 1.0 - 2.0 * s["E_deploy"] / p["E_deploy_max"]

        obs = np.concatenate([
            np.array([
                soc_n,
                tb_n,
                dep_n,
                self.prof["P_gb"][k] / self._P_SCALE,
                self.prof["v"][k] / self._V_SCALE,
                self.prof["a"][k] / self._A_SCALE,
                2.0 * lap_pos - 1.0,
                2.0 * laps_left - 1.0,
            ]),
            self._preview(k),
        ])
        return obs.astype(np.float32)

    def _bounds(self):
        s = self.state
        return feasible_P2_bounds(
            s["SoC"], s["T_bat"], s["E_deploy"], self.prof["P_gb"][s["k"]],
            self.params, self.dt, self.coef,
            allow_ice_charging=self.cfg.allow_ice_charging,
        )

    # -- gym API -----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c, p = self.cfg, self.params

        lap_ids = np.unique(self.prof["lap"])
        self._n_laps = len(lap_ids)
        self._lap_start, self._lap_len = [], []
        for lid in lap_ids:
            idx = np.flatnonzero(self.prof["lap"] == lid)
            self._lap_start.append(idx[0])
            self._lap_len.append(len(idx))
        self._lap_of = np.searchsorted(np.asarray(self._lap_start), np.arange(self.N), "right") - 1

        soc0 = c.soc_init + (self._rng.uniform(-c.soc_init_jitter, c.soc_init_jitter)
                             if c.soc_init_jitter > 0 else 0.0)
        soc0 = float(np.clip(soc0, p["SoC_min"] + 0.02, p["SoC_max"] - 0.02))
        tb0 = c.tbat_init + (self._rng.uniform(-c.tbat_init_jitter, c.tbat_init_jitter)
                             if c.tbat_init_jitter > 0 else 0.0)
        if c.cooling_factor_range is not None:
            cool = float(self._rng.uniform(*c.cooling_factor_range))
        else:
            cool = float(c.cooling_factor)

        self.state = dict(k=0, SoC=soc0, T_bat=float(tb0), E_deploy=0.0,
                          E_recharge=0.0, lap_idx=0, P2_prev=0.0, cooling=cool)
        self._soc_target = c.soc_target if c.soc_target is not None else soc0

        self.episode = dict(fuel_g=0.0, shortfall_MJ=0.0, n_shortfall=0,
                            thermal_binding=0, soc_lap_end=[], reward=0.0,
                            cost=0.0, T_bat_max=float(tb0))
        return self._obs(), {"soc_target": self._soc_target, "cooling": cool}

    def step(self, action):
        s, p, c = self.state, self.params, self.cfg
        k = s["k"]
        P_gb = float(self.prof["P_gb"][k])

        # --- 1. action -> physical command, through the safety layer -------
        lb, ub = self._bounds()
        a = float(np.asarray(action).ravel()[0])
        if c.safety_layer:
            P2_cmd = project(a, lb, ub, c.action_mode)
            projected = False
        else:
            P2_cmd = project(a, p["P_MGU_min"], p["P_MGU_max"], "relative")
            P2_clip = float(np.clip(P2_cmd, lb, ub))
            projected = abs(P2_clip - P2_cmd) > 1e-6
            P2_cmd = P2_clip

        # thermal derating is *binding* when the derated ceiling is what caps
        # the request, not merely when the temperature is above the threshold
        # A thermal event is counted as *binding* only when the derated
        # ceiling is the constraint that is actually active, i.e. the command
        # sits on the upper bound while derate < 1. Counting every step above
        # the derating threshold instead would report a temperature statistic,
        # not a constraint statistic.
        derate = thermal_derating_factor(s["T_bat"], p)
        thermal_binding = (derate < 1.0) and (P2_cmd >= ub - 1.0) and (ub > 0.0)

        # --- 2. plant ------------------------------------------------------
        (P_ICE, P2, P_mech_MGU, shortfall,
         E_dep_next, E_rec_next, m_dot) = powertrain(
            P_gb, P2_cmd, s["E_deploy"], s["E_recharge"],
            p, self.dt, control_mode="P2", Tbat_k=s["T_bat"])

        SoC_next, U2, I2 = battery_step(s["SoC"], P2, self.coef, s["T_bat"], p, self.dt)
        T_next = thermal_model(I2, s["T_bat"], p, self.dt, cooling_factor=s["cooling"])

        # consistency check: with the safety layer on, the plant must not have
        # silently clipped anything on top of our projection
        silent_clip = c.safety_layer and abs(P2 - P2_cmd) > 1e-3

        # --- 3. reward (gram-equivalent units) -----------------------------
        fuel_g = float(m_dot) * self.dt * 1000.0
        short_MJ = max(0.0, float(shortfall)) * self.dt / 1e6
        rate_MW = abs(P2 - s["P2_prev"]) / 1e6

        cost = -(fuel_g
                 + c.w_shortfall * short_MJ
                 + c.w_rate * rate_MW)

        # --- 4. lap boundary: SoC repeatability ----------------------------
        k_next = k + 1
        lap_ended = (k_next >= self.N) or (self._lap_of[min(k_next, self.N - 1)] != s["lap_idx"])
        if lap_ended:
            h = max(0.0, self._soc_target - SoC_next)
            cost -= (c.w_soc_lin * h + c.w_soc_quad * h * h)
            self.episode["soc_lap_end"].append(float(SoC_next))
            E_dep_next = 0.0          # regulatory per-lap reset
            E_rec_next = 0.0

        reward = cost * c.reward_scale

        # --- 5. bookkeeping ------------------------------------------------
        self.episode["fuel_g"] += fuel_g
        self.episode["shortfall_MJ"] += short_MJ
        self.episode["n_shortfall"] += int(short_MJ > 1e-9)
        self.episode["thermal_binding"] += int(thermal_binding)
        self.episode["reward"] += reward
        self.episode["cost"] += short_MJ
        self.episode["T_bat_max"] = max(self.episode["T_bat_max"], float(T_next))

        s.update(k=k_next, SoC=float(SoC_next), T_bat=float(T_next),
                 E_deploy=float(E_dep_next), E_recharge=float(E_rec_next),
                 P2_prev=float(P2))
        if k_next < self.N:
            s["lap_idx"] = int(self._lap_of[k_next])

        terminated = False
        if c.terminate_on_violation and (SoC_next <= p["SoC_min"] + 1e-6
                                         or T_next >= p["T_bat_safe_max"]):
            terminated = True
        truncated = k_next >= self.N

        info = dict(cost=short_MJ, P2=float(P2), P_ICE=float(P_ICE),
                    P_mech_MGU=float(P_mech_MGU), shortfall_W=float(shortfall),
                    m_dot=float(m_dot), SoC=float(SoC_next), T_bat=float(T_next),
                    E_deploy=float(E_dep_next), lb=lb, ub=ub,
                    derate=float(derate), thermal_binding=bool(thermal_binding),
                    projected=bool(projected), silent_clip=bool(silent_clip),
                    lap_ended=bool(lap_ended), fuel_g=fuel_g)
        if terminated or truncated:
            info["episode"] = self.summary()

        obs = self._obs() if k_next < self.N else np.zeros(self.observation_space.shape[0], np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -- reporting ---------------------------------------------------------

    def summary(self):
        """Episode metrics, including the SoC-equalised fuel figure.

        Comparing raw fuel between controllers that end at different SoC is
        meaningless: a controller can always look frugal by arriving with an
        empty pack. The equalised figure converts the terminal SoC deviation
        into the fuel that would have been needed to produce it, using the
        same Willans efficiency as the plant:

            fuel_eq = fuel + (SoC_target - SoC_final) * E_pack / (eta_ICE*LHV)
        """
        p = self.params
        e = dict(self.episode)
        d_soc = self._soc_target - self.state["SoC"]
        g_per_soc = p["E_pack_capacity"] / (p["eta_ICE"] * p["LHV"]) * 1000.0
        e["SoC_final"] = float(self.state["SoC"])
        e["SoC_target"] = float(self._soc_target)
        e["fuel_eq_g"] = e["fuel_g"] + d_soc * g_per_soc
        e["g_per_unit_SoC"] = g_per_soc
        return e
