"""
Evaluation harness shared by every controller (RL and baselines).

Reports, in this order of importance:

    1. fuel_eq_g      fuel at equalised terminal SoC  <- the only fair fuel figure
    2. shortfall_MJ   total unmet gearbox energy
    3. thermal_binding number of steps where the derated ceiling was the active
                      constraint
    4. gap_vs_dp      relative distance from the clairvoyant DP optimum
    5. spread across seeds
    6. inference time per step

Item 1 exists because comparing raw fuel across controllers that finish at
different SoC is not a comparison at all: any controller can look frugal by
arriving with an empty pack. Item 6 exists because it is the argument in favour
of RL: the DP is optimal but needs the whole future trajectory and an offline
solve, whereas a trained policy is a forward pass.
"""

from __future__ import annotations

import time
import numpy as np


def rollout(env, policy, seed=None, record=False):
    """
    Run one deterministic episode.

    `policy` is any callable (env, obs) -> action in [-1, 1].
    """
    obs, _ = env.reset(seed=seed)
    traj = {k: [] for k in
            ("t", "P_gb", "P2", "P_ICE", "P_mech_MGU", "SoC", "T_bat",
             "shortfall_W", "m_dot", "E_deploy", "action", "lb", "ub", "derate")}
    n, t_infer = 0, 0.0
    while True:
        t0 = time.perf_counter()
        a = policy(env, obs)
        t_infer += time.perf_counter() - t0
        k = env.state["k"]
        obs, r, term, trunc, info = env.step(a)
        if record:
            traj["t"].append(env.prof["t"][k])
            traj["P_gb"].append(env.prof["P_gb"][k])
            traj["action"].append(float(np.asarray(a).ravel()[0]))
            for key in ("P2", "P_ICE", "P_mech_MGU", "SoC", "T_bat",
                        "shortfall_W", "m_dot", "E_deploy", "lb", "ub", "derate"):
                traj[key].append(info[key])
        n += 1
        if term or trunc:
            break
    out = info["episode"]
    out["steps"] = n
    out["t_infer_us_per_step"] = 1e6 * t_infer / max(1, n)
    if record:
        out["traj"] = {k: np.asarray(v) for k, v in traj.items() if len(v)}
    return out


def evaluate(env_fn, policy, seeds=(0, 1, 2), record_first=False):
    """Run several seeds and aggregate mean / std of the headline metrics."""
    runs = []
    for i, s in enumerate(seeds):
        env = env_fn()
        runs.append(rollout(env, policy, seed=s, record=record_first and i == 0))
    keys = ("fuel_g", "fuel_eq_g", "shortfall_MJ", "n_shortfall",
            "thermal_binding", "SoC_final", "T_bat_max",
            "reward", "t_infer_us_per_step")
    agg = {k: (float(np.mean([r[k] for r in runs])),
               float(np.std([r[k] for r in runs]))) for k in keys}
    agg["_runs"] = runs
    return agg


def compare(results: dict, reference: str | None = None):
    """
    Format a comparison table. `results` maps a controller name to the dict
    returned by `evaluate`. If `reference` names one of them (typically the DP),
    a relative gap column is added.
    """
    cols = [("fuel_eq_g", "fuel_eq [g]", "{:9.2f}"),
            ("fuel_g", "fuel [g]", "{:9.2f}"),
            ("shortfall_MJ", "short [MJ]", "{:10.4f}"),
            ("thermal_binding", "therm", "{:6.0f}"),
            ("SoC_final", "SoC_f", "{:6.3f}"),
            ("t_infer_us_per_step", "us/step", "{:8.1f}")]

    head = f"{'controller':<16}" + "".join(f"{c[1]:>12}" for c in cols)
    if reference is not None:
        head += f"{'gap vs ' + reference:>16}"
    lines = [head, "-" * len(head)]

    ref = results.get(reference, {}).get("fuel_eq_g", (None,))[0] if reference else None
    for name, r in results.items():
        row = f"{name:<16}"
        for key, _, fmt in cols:
            row += f"{fmt.format(r[key][0]):>12}"
        if ref:
            row += f"{100.0 * (r['fuel_eq_g'][0] - ref) / ref:>15.2f}%"
        lines.append(row)
    return "\n".join(lines)


def save_trajectory(path, run):
    """Persist a recorded rollout for plotting in the notebooks."""
    traj = run.get("traj")
    if traj is None:
        raise ValueError("rollout was not recorded (record=True needed)")
    meta = {k: v for k, v in run.items() if k != "traj" and np.isscalar(v)}
    np.savez_compressed(path, **traj, **{f"meta_{k}": v for k, v in meta.items()})
    return path
