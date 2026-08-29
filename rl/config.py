"""
Configuration objects for the RL layer.

Everything that can change the numbers lives here, so an experiment is fully
described by (EnvConfig, SACConfig, TrainConfig) + seed. No magic constants
buried in the training loop.

Units convention used throughout the RL layer
---------------------------------------------
The reward is expressed in *gram-equivalents of fuel*. Every penalty term is
converted into the grams of fuel that would have been needed to produce the
same effect, so the reward is directly interpretable and the weights are
physically anchored instead of hand-tuned. See rl/env.py for the derivations.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import json


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    # --- profile -----------------------------------------------------------
    profile: str = "multi_lap"          # "single_lap" | "multi_lap"
    dt: float = 0.1                     # [s] telemetry sampling period
    n_laps: Optional[int] = None        # truncate the multi-lap profile to N laps
                                        # (curriculum: 1 -> 3 -> 5)

    # --- observation -------------------------------------------------------
    preview_horizon: int = 5            # number of look-ahead samples of P_gb
    preview_stride: int = 5             # spacing between preview samples [steps]
                                        # 5 * 0.1 s = 0.5 s apart -> 2.5 s horizon
    # preview_horizon = 0 turns the problem into a POMDP (see RL_DESIGN.md §3.2)

    # --- initial conditions / domain randomization -------------------------
    soc_init: float = 0.8
    soc_init_jitter: float = 0.0        # uniform +/- jitter on SoC_0
    tbat_init: float = 40.0             # [degC]
    tbat_init_jitter: float = 0.0       # uniform +/- jitter on T_bat,0
    cooling_factor: float = 1.0         # 1.0 = nominal cooling
    cooling_factor_range: Optional[Tuple[float, float]] = None
                                        # if set, sampled uniformly at reset

    # --- SoC repeatability target -----------------------------------------
    soc_target: Optional[float] = None  # defaults to the *realised* soc_init

    # --- safety layer ------------------------------------------------------
    safety_layer: bool = True           # project the action onto the feasible set
    action_mode: str = "relative"       # "relative" | "absolute"
    allow_ice_charging: bool = True     # MGU-K may act as generator under traction

    # --- reward weights (all in gram-equivalent units) ---------------------
    w_shortfall: float = 500.0          # [g / MJ] of unmet gearbox energy
    w_rate: float = 1.0                 # [g / MW] of step-to-step P2 change
    w_soc_lin: float = 260.0            # [g / unit SoC] deficit at lap end
    w_soc_quad: float = 20000.0         # [g / (unit SoC)^2] deficit at lap end
    reward_scale: float = 0.05          # network conditioning only, not physics

    # --- termination -------------------------------------------------------
    terminate_on_violation: bool = False  # only meaningful with safety_layer=False

    # --- data paths --------------------------------------------------------
    data_root: str = "."

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class SACConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    gamma: float = 0.995                # near 1: finite-horizon economic cost
    tau: float = 0.005                  # Polyak averaging coefficient
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000

    # --- entropy -----------------------------------------------------------
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: Optional[float] = None   # default: -dim(A)

    # --- TD3-style tricks kept in SAC --------------------------------------
    policy_delay: int = 1               # SAC normally 1; >1 mimics TD3

    # --- CMDP / Lagrangian -------------------------------------------------
    use_lagrangian: bool = False        # constrained formulation (RL_DESIGN.md §5)
    cost_limit: float = 0.05            # [MJ] budget of shortfall energy / episode
    lr_lambda: float = 1e-3
    lambda_init: float = 1.0
    lambda_max: float = 1000.0
    
    actor_start_it: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    warmup_steps: int = 5_000           # uniform random actions
    updates_per_step: int = 1           # UTD ratio
    eval_every: int = 10_000
    eval_episodes: int = 3
    seed: int = 0
    out_dir: str = "runs/sac"
    save_every: int = 50_000
    bc_checkpoint: Optional[str] = None  # warm-start actor from behaviour cloning
    log_every: int = 1_000

    def to_dict(self):
        return asdict(self)


def dump_configs(path, **cfgs):
    """Write every config to a single JSON next to the run outputs."""
    payload = {k: (v.to_dict() if hasattr(v, "to_dict") else v)
               for k, v in cfgs.items()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload
