# `rl/` — Reinforcement learning layer

Soft Actor-Critic energy management for the hybrid power unit, built on the
existing quasistatic plant. The DP controller is the clairvoyant optimum and is
used as the benchmark, not as a competitor: the RL agent solves the same Bellman
equation by approximation, from the current state only, with a short preview
window.

## Layout

| file | role |
|---|---|
| `config.py` | every tunable in one place: `EnvConfig`, `SACConfig`, `TrainConfig` |
| `env.py` | `EMSEnv`, a Gymnasium environment wrapping `plant/` |
| `safety.py` | analytic feasible set for the electrical command; action projection |
| `baselines.py` | rule-based, ECMS and DP-replay policies, run inside the same env |
| `buffer.py` | replay buffer, optional prioritisation, cost channel for CMDP |
| `sac.py` | SAC with automatic entropy tuning and optional Lagrangian constraint |
| `bc.py` | behaviour cloning warm-start from a teacher controller |
| `train.py` | curriculum training loop with domain randomisation |
| `evaluate.py` | rollouts, SoC-equalised metrics, comparison table |

## Quick start

```bash
pip install torch gymnasium            # in addition to requirements.txt

# reduced budget, single lap, ~2 minutes on CPU
python scripts/run_rl_experiment.py --steps 20000 --profile single_lap --out runs/quick

# full curriculum, five laps
python scripts/run_rl_experiment.py --steps 300000 --out runs/sac

# constrained (CMDP) formulation: shortfall becomes a budget, not a penalty
python scripts/run_rl_experiment.py --steps 300000 --lagrangian --cost-limit 0.05

# clone from an exported DP trajectory instead of ECMS
python scripts/run_rl_experiment.py --dp-traj data/results/dp_P2.npy
```

Tests: `python -m pytest tests/test_rl.py -q`

## MDP formulation

**State** (`8 + preview_horizon` components, analytically normalised)

`SoC`, `T_bat`, per-lap deploy energy remaining, current `P_gb`, `v`, `a`,
normalised position in the lap, laps remaining, and `preview_horizon` future
samples of `P_gb`.

`T_bat` and the deploy accumulator are not optional: without them the transition
is not Markov. Lap phase and laps-remaining are needed because the problem is
finite-horizon and non-stationary — the optimal policy depends on how much race
is left, which is exactly why the unconstrained multi-lap DP empties the pack in
the first two laps.

**Action** — scalar in `[-1, 1]`, mapped by the safety layer onto the currently
admissible interval of battery terminal power `P2` (the electrical domain, where
the 350 kW regulatory limit and the deploy budget are defined). Two mappings are
provided, `relative` (default) and `absolute`; the trade-off is documented in
`safety.py`.

**Reward** — expressed in gram-equivalents of fuel, so every weight is
physically anchored rather than hand-tuned:

```
r = -( fuel_g
     + w_shortfall * unmet_MJ        # default 500 g/MJ (natural price: 45.6 g/MJ)
     + w_rate      * |ΔP2| in MW )   # suppresses chattering
    - lap-end hinge on SoC           # 260 g per unit SoC + quadratic term
```

The shortfall penalty is **continuous** in the shortfall magnitude. A binary
large constant makes the optimiser indifferent between 0.1 MJ and 12 MJ of unmet
demand and provides no gradient.

**Constraints** — `SoC ∈ [0.2, 0.9]`, `T_bat` below the safety limit, per-lap
deploy budget and the MGU-K envelope are hard (Level III) and are enforced by
construction in `safety.py`, so the agent never has to learn them. End-of-lap SoC
repeatability is soft (Level I), as in the DP. Shortfall can be either a reward
term or an explicit budget via `--lagrangian`.

## Reported metrics

`fuel_eq_g` is the headline figure: raw fuel corrected for the terminal SoC
deviation at the plant's own Willans efficiency. Comparing raw fuel between
controllers that finish at different SoC is not a comparison — any controller can
look frugal by arriving with an empty pack.

Also reported: total shortfall, number of steps where the thermal ceiling was the
active constraint, terminal SoC, spread across seeds, and inference time per
step.

## Two findings about the plant

1. `plant/battery.py` applies the thermal derating factor to the *battery* power
   ceiling `Uoc²/(4·R_int)`. With `R_int = 0.01 Ω` that ceiling is 1.8–3.0 MW
   across the SoC window, so the derated value only drops below the 350 kW MGU-K
   limit above roughly 58 °C — one degree short of the hard safety limit. As
   coded, the derating is very nearly inert. The README describes the intent
   differently and more plausibly (derating on MGU-K maximum discharge power);
   `safety.py` implements that intent, which is what makes the thermal constraint
   observable and binding at realistic temperatures.

2. Running the rule-based controller inside `EMSEnv` reproduces the canonical
   figures (773.5 g, 2.02 MJ, `SoC_f` 0.5417) to within ~1%, which validates the
   harness. Residual differences come from the safety layer bounding the command
   before the plant sees it, where the notebooks let `battery_step()` clip
   internally.
