"""
Training loop.

Structure follows the sample-efficiency recipe that the process-control
literature converges on, adapted to this plant:

  stage 0  behaviour cloning from the DP policy            (optional, rl/bc.py)
  stage 1  SAC on a single lap, randomised initial state
  stage 2  SAC on three laps
  stage 3  SAC on five laps with randomised cooling effectiveness

Each stage inherits the weights and the replay buffer of the previous one. The
curriculum exists because the credit-assignment horizon grows by a factor of
five between stage 1 and stage 3, and because the single-lap and multi-lap
problems are genuinely related tasks - which is precisely the condition under
which transfer helps.

Domain randomisation (initial SoC, initial temperature, cooling effectiveness)
serves two purposes. It is the standard remedy for overfitting to one
trajectory, and here it also guarantees that the thermal constraint is actually
exercised: at nominal cooling the derating ceiling binds rarely, so an agent
trained only there never learns to manage it. Sampling the cooling factor over
the same range already used in the parametric stress test turns that stress
test into a task distribution.

The loop logs a CSV that can be read directly in the notebooks.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import replace

import numpy as np
import torch

from .config import EnvConfig, SACConfig, TrainConfig, dump_configs
from .env import EMSEnv, load_profile
from .buffer import ReplayBuffer
from .sac import SAC
from .evaluate import evaluate


DEFAULT_CURRICULUM = [
    dict(name="lap1", steps=100_000,
         env=dict(profile="single_lap", soc_init_jitter=0.05, tbat_init_jitter=5.0)),
    dict(name="lap3", steps=100_000,
         env=dict(profile="multi_lap", n_laps=3, soc_init_jitter=0.05,
                  tbat_init_jitter=5.0)),
    dict(name="lap5", steps=100_000,
         env=dict(profile="multi_lap", n_laps=None, soc_init_jitter=0.05,
                  tbat_init_jitter=5.0, cooling_factor_range=(0.3, 1.0))),
]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env_fn(base_cfg: EnvConfig, overrides: dict, cache={}):
    """Env factory with a profile cache, so the telemetry is parsed once per
    configuration instead of once per episode."""
    cfg = replace(base_cfg, **overrides)
    key = (cfg.profile, cfg.n_laps, cfg.data_root)
    if key not in cache:
        cache[key] = load_profile(cfg)
    prof = cache[key]

    def _fn():
        return EMSEnv(cfg, profile=prof)
    return _fn, cfg


def train(env_cfg: EnvConfig | None = None,
          sac_cfg: SACConfig | None = None,
          train_cfg: TrainConfig | None = None,
          curriculum=None,
          prioritized=False):
    env_cfg = env_cfg or EnvConfig()
    sac_cfg = sac_cfg or SACConfig()
    train_cfg = train_cfg or TrainConfig()
    curriculum = curriculum if curriculum is not None else DEFAULT_CURRICULUM

    os.makedirs(train_cfg.out_dir, exist_ok=True)
    dump_configs(os.path.join(train_cfg.out_dir, "config.json"),
                 env=env_cfg, sac=sac_cfg, train=train_cfg,
                 curriculum=curriculum, prioritized=prioritized)
    set_seed(train_cfg.seed)

    probe_fn, _ = make_env_fn(env_cfg, curriculum[0]["env"])
    probe = probe_fn()
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]

    agent = SAC(obs_dim, act_dim, sac_cfg)
    if train_cfg.bc_checkpoint:
        agent.load(train_cfg.bc_checkpoint, actor_only=True)
        print(f"[train] actor warm-started from {train_cfg.bc_checkpoint}")

    buf = ReplayBuffer(obs_dim, act_dim, sac_cfg.buffer_size, prioritized=prioritized)

    log_path = os.path.join(train_cfg.out_dir, "log.csv")
    log_f = open(log_path, "w", newline="")
    logger = csv.writer(log_f)
    logger.writerow(["stage", "step", "episode", "ep_return", "fuel_g",
                     "fuel_eq_g", "shortfall_MJ", "thermal_binding",
                     "SoC_final", "alpha", "lambda", "loss_q", "wall_s"])

    global_step, episode = 0, 0
    t_start = time.time()
    info_upd = {}

    for stage in curriculum:
        env_fn, stage_cfg = make_env_fn(env_cfg, stage["env"])
        env = env_fn()
        obs, _ = env.reset(seed=train_cfg.seed + episode)
        stage_steps = int(stage["steps"])
        print(f"\n=== stage {stage['name']}  ({stage_steps} steps, "
              f"{env.N} steps/episode) ===")

        for _ in range(stage_steps):
            if global_step < train_cfg.warmup_steps and train_cfg.bc_checkpoint is None:
                a = env.action_space.sample()
            else:
                a = agent.act(obs, deterministic=False)

            nobs, r, term, trunc, info = env.step(a)
            # `done` must flag only *true* terminations. Truncation at the end
            # of the profile is an artificial time limit: bootstrapping through
            # it is correct, treating it as terminal is not.
            buf.add(obs, a, r, nobs, float(term), cost=info["cost"])
            obs = nobs
            global_step += 1

            if len(buf) >= max(train_cfg.warmup_steps, sac_cfg.batch_size):
                for _ in range(train_cfg.updates_per_step):
                    batch = buf.sample(sac_cfg.batch_size)
                    info_upd = agent.update(batch)
                    if prioritized:
                        buf.update_priorities(batch["idx"], info_upd["td_err"])

            if term or trunc:
                ep = info["episode"]
                episode += 1
                lam_info = agent.update_lambda(ep["cost"]) if sac_cfg.use_lagrangian else None
                logger.writerow([stage["name"], global_step, episode,
                                 round(ep["reward"], 3), round(ep["fuel_g"], 2),
                                 round(ep["fuel_eq_g"], 2), round(ep["shortfall_MJ"], 5),
                                 ep["thermal_binding"], round(ep["SoC_final"], 4),
                                 round(info_upd.get("alpha", float(agent.alpha)), 4),
                                 round(lam_info["lam"], 4) if lam_info else 0.0,
                                 round(info_upd.get("loss_q", float("nan")), 4),
                                 round(time.time() - t_start, 1)])
                log_f.flush()
                if episode % 5 == 0:
                    print(f"[{stage['name']}] step {global_step:7d} ep {episode:4d} "
                          f"fuel_eq {ep['fuel_eq_g']:8.1f} g  "
                          f"short {ep['shortfall_MJ']:7.4f} MJ  "
                          f"SoC_f {ep['SoC_final']:.3f}  R {ep['reward']:9.1f}")
                obs, _ = env.reset(seed=train_cfg.seed + episode)

            if global_step % train_cfg.eval_every == 0:
                res = evaluate(env_fn, agent.policy(deterministic=True),
                               seeds=tuple(range(train_cfg.eval_episodes)))
                print(f"  [eval @ {global_step}] fuel_eq "
                      f"{res['fuel_eq_g'][0]:.1f} +/- {res['fuel_eq_g'][1]:.1f} g | "
                      f"short {res['shortfall_MJ'][0]:.4f} MJ")

            if global_step % train_cfg.save_every == 0:
                agent.save(os.path.join(train_cfg.out_dir, f"sac_{global_step}.pt"))

    agent.save(os.path.join(train_cfg.out_dir, "sac_final.pt"))
    log_f.close()
    print(f"\n[train] done in {time.time() - t_start:.0f}s -> {train_cfg.out_dir}")
    return agent


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None,
                    help="override the total number of steps, split evenly "
                         "across the curriculum stages")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/sac")
    ap.add_argument("--lagrangian", action="store_true",
                    help="constrained (CMDP) formulation: shortfall becomes a "
                         "budget instead of a reward penalty")
    ap.add_argument("--cost-limit", type=float, default=0.05, help="[MJ]")
    ap.add_argument("--prioritized", action="store_true")
    ap.add_argument("--bc", type=str, default=None, help="behaviour-cloning checkpoint")
    ap.add_argument("--data-root", type=str, default=".")
    args = ap.parse_args()

    env_cfg = EnvConfig(data_root=args.data_root)
    if args.lagrangian:
        # in the constrained formulation the shortfall is a constraint, not a
        # reward term: keeping both would price it twice
        env_cfg = replace(env_cfg, w_shortfall=0.0)

    sac_cfg = SACConfig(use_lagrangian=args.lagrangian, cost_limit=args.cost_limit)
    train_cfg = TrainConfig(seed=args.seed, out_dir=args.out, bc_checkpoint=args.bc)

    curriculum = DEFAULT_CURRICULUM
    if args.steps:
        per = args.steps // len(curriculum)
        curriculum = [dict(s, steps=per) for s in curriculum]

    train(env_cfg, sac_cfg, train_cfg, curriculum, prioritized=args.prioritized)
