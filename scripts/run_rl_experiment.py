"""
End-to-end RL experiment.

Runs, in order:
    1. the baselines (rule-based, ECMS) inside the RL environment, so that every
       controller is measured by the same harness;
    2. behaviour cloning of the ECMS (or of a DP trajectory, if one is supplied)
       to warm-start the actor;
    3. SAC training through the curriculum;
    4. the comparison table and the diagnostic figure.

Usage
-----
    python scripts/run_rl_experiment.py --steps 300000
    python scripts/run_rl_experiment.py --steps 20000 --profile single_lap   # quick
    python scripts/run_rl_experiment.py --lagrangian --cost-limit 0.05
    python scripts/run_rl_experiment.py --dp-traj data/results/dp_P2.npy

Every number printed is produced by this script; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

# make the repository root importable when the script is launched directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--profile", choices=["single_lap", "multi_lap"], default="multi_lap")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/experiment")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--lagrangian", action="store_true")
    ap.add_argument("--cost-limit", type=float, default=0.05)
    ap.add_argument("--prioritized", action="store_true")
    ap.add_argument("--bc-epochs", type=int, default=50)
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--dp-traj", default=None,
                    help="optional .npy of DP electrical commands P2[k] [W]; "
                         "used as the cloning teacher and as the reference row")
    ap.add_argument("--cooling", type=float, default=1.0)
    args = ap.parse_args()

    from rl.config import EnvConfig, SACConfig, TrainConfig
    from rl.env import EMSEnv, load_profile
    from rl.baselines import RuleBasedPolicy, ECMSPolicy, ReplayPolicy
    from rl.evaluate import evaluate, compare, rollout, save_trajectory
    from rl.bc import collect_demonstrations, clone
    from rl.sac import SAC
    from rl.train import train, DEFAULT_CURRICULUM

    os.makedirs(args.out, exist_ok=True)
    env_cfg = EnvConfig(profile=args.profile, data_root=args.data_root,
                        cooling_factor=args.cooling)
    if args.lagrangian:
        env_cfg = replace(env_cfg, w_shortfall=0.0)
    prof = load_profile(env_cfg)
    mk = lambda: EMSEnv(env_cfg, profile=prof)

    # ---- 1. baselines ----------------------------------------------------
    print("== baselines ==")
    results = {
        "rule_based": evaluate(mk, RuleBasedPolicy(), seeds=tuple(args.seeds)),
        "ecms": evaluate(mk, ECMSPolicy(), seeds=tuple(args.seeds)),
        "ecms_rate": evaluate(mk, ECMSPolicy(rate_penalty=1e-3), seeds=tuple(args.seeds)),
    }
    teacher = ECMSPolicy()
    reference = None
    if args.dp_traj:
        dp = np.load(args.dp_traj)
        teacher = ReplayPolicy(dp)
        results["dp"] = evaluate(mk, teacher, seeds=tuple(args.seeds))
        reference = "dp"
    print(compare(results, reference=reference))

    # ---- 2. behaviour cloning -------------------------------------------
    probe = mk()
    obs_dim = probe.observation_space.shape[0]
    sac_cfg = SACConfig(use_lagrangian=args.lagrangian, cost_limit=args.cost_limit, actor_start_it=0 if args.no_bc else 5000)
    bc_ckpt = None
    if not args.no_bc:
        print("\n== behaviour cloning ==")
        O, A = collect_demonstrations(mk, teacher, seeds=[0])
        agent = SAC(obs_dim, 1, sac_cfg)
        clone(agent, O, A, epochs=args.bc_epochs)
        bc_ckpt = os.path.join(args.out, "bc_actor.pt")
        agent.save(bc_ckpt)
        results["bc_actor"] = evaluate(mk, agent.policy(True), seeds=tuple(args.seeds))

    # ---- 3. SAC ----------------------------------------------------------
    print("\n== SAC ==")
    curriculum = DEFAULT_CURRICULUM
    if args.profile == "single_lap":
        curriculum = [dict(name="lap1", steps=args.steps,
                           env=dict(profile="single_lap", soc_init_jitter=0.05,
                                    tbat_init_jitter=5.0))]
    else:
        per = args.steps // len(curriculum)
        curriculum = [dict(s, steps=per) for s in curriculum]

    train_cfg = TrainConfig(seed=args.seeds[0], out_dir=args.out,
                            bc_checkpoint=bc_ckpt,
                            eval_every=max(5000, args.steps // 10))
    agent = train(env_cfg, sac_cfg, train_cfg, curriculum, prioritized=args.prioritized)
    results["sac"] = evaluate(mk, agent.policy(True), seeds=tuple(args.seeds),
                              record_first=True)

    # ---- 4. report -------------------------------------------------------
    print("\n== final comparison ==")
    table = compare(results, reference=reference)
    print(table)
    with open(os.path.join(args.out, "comparison.txt"), "w") as f:
        f.write(table + "\n")

    run = results["sac"]["_runs"][0]
    save_trajectory(os.path.join(args.out, "sac_trajectory.npz"), run)
    try:
        plot(run, os.path.join(args.out, "sac_trajectory.png"))
    except Exception as e:                                  # matplotlib optional
        print(f"[plot] skipped: {e}")
    print(f"\nartifacts in {args.out}")


def plot(run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = run["traj"]
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    ax[0].plot(tr["t"], tr["P_gb"] / 1e3, lw=0.8, label="demand at gearbox")
    ax[0].plot(tr["t"], tr["P_ICE"] / 1e3, lw=0.8, label="ICE")
    ax[0].plot(tr["t"], tr["P_mech_MGU"] / 1e3, lw=0.8, label="MGU-K (mech)")
    ax[0].set_ylabel("power [kW]")
    ax[0].legend(ncol=3, fontsize=8)

    ax[1].plot(tr["t"], tr["P2"] / 1e3, lw=0.8, color="k", label="P2 command")
    ax[1].fill_between(tr["t"], tr["lb"] / 1e3, tr["ub"] / 1e3, alpha=0.15,
                       label="feasible envelope")
    ax[1].set_ylabel("electrical [kW]")
    ax[1].legend(ncol=2, fontsize=8)

    ax[2].plot(tr["t"], tr["SoC"], lw=1.0, label="SoC")
    ax[2].set_ylabel("SoC [-]")
    ax2 = ax[2].twinx()
    ax2.plot(tr["t"], tr["T_bat"], lw=1.0, color="tab:red", label="T_bat")
    ax2.set_ylabel("T_bat [degC]", color="tab:red")

    ax[3].plot(tr["t"], np.maximum(0.0, tr["shortfall_W"]) / 1e3, lw=0.8,
               color="tab:orange")
    ax[3].set_ylabel("shortfall [kW]")
    ax[3].set_xlabel("time [s]")

    fig.suptitle(f"SAC policy - fuel_eq {run['fuel_eq_g']:.1f} g, "
                 f"shortfall {run['shortfall_MJ']:.3f} MJ, "
                 f"SoC_f {run['SoC_final']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
