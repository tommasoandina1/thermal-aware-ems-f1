# Thermal-Aware Energy Management for a Hybrid Race Powertrain

**How should a hybrid race car split power between its combustion engine and its
electric motor, lap after lap, when the battery is small and gets hot?** This
project answers that question on real Formula 1 telemetry, under the 2026
technical regulations, by building a physics-based model of the power unit and
comparing five control strategies — from a simple heuristic, through optimal
control with full knowledge of the lap ahead, to a reinforcement learning policy
that has to decide without it.

![Summary of results](img/results_overview.png)

[![tests](https://github.com/tommasoandina1/Thermal-Aware-Energy-Management-System-for-a-Hybrid-Motorsport-Powertrain/actions/workflows/tests.yml/badge.svg)](https://github.com/tommasoandina1/Thermal-Aware-Energy-Management-System-for-a-Hybrid-Motorsport-Powertrain/actions/workflows/tests.yml)


|  |  |
|---|---|
| **What** | Simulation and control framework for a Formula 1 hybrid power unit |
| **Role** | Sole author — modeling, control design, implementation, validation |
| **Context** | MSc thesis in Mechatronics Engineering, Politecnico di Torino |
| **Built with** | Python, NumPy, SciPy, PyTorch, Gymnasium, Docker, pytest |
| **Scale** | ~4 500 lines · 5 control strategies · 34 automated tests · 6 notebooks |
| **Status** | Complete and reproducible — `docker compose build`, then `pytest tests` |

---

## In plain terms

A modern Formula 1 car has two power sources: a combustion engine and an
electric motor fed by a battery. Thousands of times per lap, something has to
decide how much each one contributes. Deploy too eagerly and the battery is
empty before the end of the lap; deploy too conservatively and fuel is wasted.
The battery also heats up, and a hot battery is not allowed to deliver full
power — so the decision made in turn 3 constrains what is possible in turn 11.

This project builds a physics model of that power unit from first principles,
validates it against real telemetry from a qualifying lap, and uses it as a
testbed to compare five ways of making that decision:

1. **A simple rule.** Deploy in proportion to how full the battery is. Quick to
   write, and it leaves 2 MJ of the driver's power request unmet.
2. **ECMS**, which prices electricity in units of fuel and minimises the total
   at every instant. Near-optimal, and it needs no knowledge of the future.
3. **Dynamic programming**, which knows the entire lap in advance and computes
   the provably optimal split. Not implementable in a real car — it is the
   yardstick everything else is measured against.
4. **Dynamic programming with thermal awareness**, which accounts for the fact
   that the battery's power ceiling falls as it heats up.
5. **Reinforcement learning**, which learns a policy from experience and decides
   from the current state alone — the only approach here that could run on a
   real car, at roughly 80 microseconds per decision.

The interesting results are not "which one wins". They are the trade-offs that
become visible once the model is honest enough to show them.

## What the numbers say

- **Knowing the lap ahead is worth 1.8% fuel.** ECMS, which decides instant by
  instant, costs 820.09 g against 805.48 g for dynamic programming with the full
  velocity profile — at the same final charge and the same deployed energy. That
  gap is the price of not seeing the future, and it is smaller than one might
  expect.
- **The fuel-optimal strategy is not raceable.** Over five laps it drains the
  battery to its floor at every lap boundary, leaving no reserve. Adding a
  one-sided penalty on end-of-lap charge **halves the power demand the car
  cannot meet** (1.79 → 0.90 MJ) for 3.5% more fuel — and the simple fixed floor
  beats the lap-dependent schedule on both axes, contradicting the design intent
  behind the latter.
- **Thermal derating binds earlier than expected.** The working hypothesis was
  that charge runs out before the pack gets hot. It does not: derating is
  already active for 261 of 3886 stages at nominal cooling, rising to 1504 at
  30% cooling effectiveness. The 2026 pack is small enough that the current
  needed to deploy 350 kW heats it past its threshold before the charge floor is
  reached.
- **Closing the loop recovers feasibility.** Under degraded cooling, iterating
  between the optimal policy and the thermal model until they agree (11
  iterations, 10.37 → 0.29 °C residual) takes unmet demand from 12.12 MJ to
  0.11 MJ for 2.9% more fuel — without promoting temperature to a third state
  variable.

Every number above is reproduced, with the reasoning behind it, in
[`controller/Summary_of_Results.ipynb`](controller/Summary_of_Results.ipynb).

## Quickstart

```bash
git clone https://github.com/tommasoandina1/Thermal-Aware-Energy-Management-System-for-a-Hybrid-Motorsport-Powertrain
cd Thermal-Aware-Energy-Management-System-for-a-Hybrid-Motorsport-Powertrain

docker compose build
docker compose run --rm jupyter-env pytest tests -q     # 34 tests
docker compose up                                        # JupyterLab on :8889
```

Without Docker:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt -r requirements-rl.txt -r requirements-dev.txt
pytest tests -q
```

`torch` is installed separately because `--index-url` replaces PyPI rather than
adding to it. The CPU wheel is ~300 MB against ~4 GB for the default CUDA build.

Reproduce the pipeline end to end:

```bash
python scripts/build_velocity_profile.py    # telemetry -> velocity/acceleration
python scripts/simulation.py                # -> gearbox power demand
python scripts/make_results_overview.py     # -> the figure at the top of this page

python scripts/run_rl_experiment.py --steps 20000 --profile single_lap --out runs/quick
```

Then open
[`controller/Summary_of_Results.ipynb`](controller/Summary_of_Results.ipynb) for
the results. The first telemetry download populates a FastF1 cache (~230 MB)
outside the repository, at `$FASTF1_CACHE`.

## What is being modeled

A 2026-regulation hybrid power unit: a 1.6 L V6 turbo internal combustion engine
capped at 400 kW, an MGU-K electrical machine limited to ±350 kW with a 9 MJ
per-lap deploy allowance, and a 5.7 MJ energy store. At every instant the
controller decides how much of the driver's power request each one delivers. The
car follows a fixed velocity profile taken from a real qualifying lap, so lap
time is fixed and the question is purely about energy.

- **Vehicle dynamics** — longitudinal force balance (drag, rolling resistance,
  inertia) converted to power demand at the gearbox. Acceleration is
  reconstructed from raw telemetry with a quartic spline on the irregular time
  grid, which keeps the resulting power peaks inside the physical ceiling.
- **ICE** — affine Willans line (`P_e = eta_ICE * P_fuel - P_ICE0`) with the
  regulatory fuel-flow limit, calibrated so the achievable power matches the
  400 kW cap.
- **MGU-K** — instantaneous power limits, cumulative per-lap deploy budget reset
  at each lap boundary, and a temperature-dependent derating factor on maximum
  discharge power, applied on the deploy envelope where the regulatory limit
  also lives.
- **Battery** — equivalent circuit (open-circuit voltage in series with internal
  resistance), solved in closed form for terminal voltage, with Coulomb counting
  for charge integration. The OCV curve is calibrated from a published
  high-power Li-ion cell (Samsung INR21700-48X, Rukavina et al. 2023) and
  rescaled to the pack voltage window, since real pack data is proprietary.
- **Thermal** — lumped-capacitance pack temperature driven by Joule heating and
  convective cooling to a fixed-inlet coolant, with linear derating above a
  threshold.
- **Shortfall accounting** — the plant never silently compensates a saturated
  component with another. Any gap between requested and deliverable power is
  reported, for the controller to handle. This is what makes the fuel numbers
  comparable at all.

## Control strategies

| Strategy | Knowledge of the future | Status |
|---|---|---|
| Rule-based (SoC-scheduled) | None | Implemented |
| ECMS | None (DP-referenced equivalence factor) | Implemented |
| DP, single lap | Full (benchmark / upper bound) | Implemented |
| DP, multi-lap + charge repeatability constraint | Full | Implemented |
| DP + thermal derating, self-consistent | Full, thermally aware | Implemented |
| Reinforcement learning (SAC) | Learned, causal | Implemented |

The dynamic program is not a competitor to the learned policy, it is its oracle:
both solve the same Bellman equation, one exactly on a discretized grid with
full knowledge of the future, the other by approximation from the current state
and a short preview window. The reinforcement learning layer is therefore
measured by its distance from the DP optimum under the same constraints, against
an inference cost three to four orders of magnitude lower. Hard constraints —
the charge window, the per-lap deploy budget, the MGU-K envelope and its thermal
derating — are enforced analytically by a safety layer rather than learned, so
the agent optimizes inside a set that is admissible by construction. See
[`rl/README.md`](rl/README.md).

## Methods

Dynamic programming (backward value iteration on a discretized state grid,
vectorized) · ECMS (equivalent consumption minimization) · soft actor-critic
with automatic entropy tuning · constrained MDP via Lagrangian dual ascent ·
behaviour cloning warm-start · curriculum training with domain randomization ·
quasistatic powertrain modeling · equivalent-circuit battery with
lumped-capacitance thermal model · fixed-point iteration for policy–plant
self-consistency · signal reconstruction from noisy telemetry (spline
differentiation).

## Project structure

```
.
├── controller/          # the five notebooks + rule-based baseline
├── plant/               # parameters, vehicle dynamics, battery, powertrain
├── rl/                  # SAC energy management: env, safety layer, agent, baselines
├── scripts/             # telemetry -> velocity profile -> power demand -> figures
├── tests/               # 34 pytest checks on the plant model and the RL layer
├── docs/                # design rationale, study guide, thermal-derating fix
├── data/                # inputs and exported results (.npy / .npz)
├── img/                 # exported figures
├── paths.py             # single source of truth for every path
└── Compare_Controllers.ipynb
```

The reasoning behind every design decision in the RL layer — the choice of state
and action, why the constraints live in the admissible set rather than in the
reward, the gram-equivalent reward units and their derivations — is documented
in [`docs/SCELTE_RL.md`](docs/SCELTE_RL.md) *(in Italian)*.

## Scope and honest limitations

Deliberate simplifications, stated rather than hidden:

- No turbocharger sub-model and no engine-speed dependence (the Willans-line
  simplification validated in Ebbesen et al.).
- No hydraulic-brake dissipation model; unrecovered braking is reported as
  shortfall.
- MGU-K power is controlled in electrical terms in the DP/ECMS formulations
  (`eta_MGU = 1` in the power balance) while the forward pass uses the real
  plant, so the resulting mismatch is measured rather than assumed away.
- Thermal derating is enforced as a bound made self-consistent by fixed-point
  iteration, not by promoting temperature to a third DP state — a deliberate
  cost/accuracy tradeoff, discussed in
  [`controller/04_multi_lap_DP_thermal.ipynb`](controller/04_multi_lap_DP_thermal.ipynb).
- The battery safety limit is not a hard constraint. `T_bat_safe_max` is the
  temperature at which the derating factor reaches zero, so deployment stops
  there — but regeneration continues to dissipate `I²R` into the pack and
  nothing prevents the temperature from exceeding it. At 30% cooling
  effectiveness the simulation reaches 61.3 °C against a 60 °C limit. Enforcing
  it would require derating recovery as well, which the current model does not
  do.
- Internal resistance is constant (not temperature-dependent); OCV and thermal
  parameters come from published reference data, not from a calibrated pack.
- The ECMS equivalence factor tracks the DP charge trajectory in the notebooks,
  so ECMS there is an upper bound on achievable performance rather than a causal
  controller. A causal variant, with proportional feedback on charge, runs
  inside the RL harness (`rl/baselines.py`) and is the one used for
  controller-to-controller comparison.

## References

- Guzzella, L., Sciarretta, A. *Vehicle Propulsion Systems*, 3rd ed., Springer, 2013.
- Ebbesen, S., Salazar, M., Elbert, P., Bussi, C., Onder, C.H. "Time-Optimal
  Control Strategies for a Hybrid Electric Race Car." *IEEE Transactions on
  Control Systems Technology*, 26(1), 2018.
- Rukavina, F., Leko, D., Matijašić, M., Bralić, I., Ugalde, J.M., Vašak, M.
  "Identification of equivalent circuit model parameters for a Li-ion battery
  cell." *Proc. 2023 IEEE 11th International Conference on Systems and Control.*
- Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. "Soft Actor-Critic: Off-Policy
  Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML*, 2018.

## License

MIT — see [LICENSE](LICENSE).
