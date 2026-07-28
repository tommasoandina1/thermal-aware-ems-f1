# Thermal-Aware Energy Management System for a Hybrid Motorsport Powertrain

A quasistatic simulation and control framework for the energy management strategy (EMS) of a hybrid motorsport powertrain (F1 2026 / LMDh-inspired), built from real qualifying telemetry. The project benchmarks a rule-based baseline against Equivalent Consumption Minimization Strategy (ECMS) and Dynamic Programming (DP) controllers over a full 5-lap race, then extends the DP formulation to account for per-lap repeatability and thermal derating.

## Key results

- Identified and quantified a fuel-optimal but non-repeatable baseline SoC strategy: the DP fully depletes reserve within the first two laps, causing a measurable power shortfall in the closing laps of the race.
- Corrected repeatability with an asymmetric soft constraint on end-of-lap SoC (one-sided hinge penalty), compared against a fixed-target and a lap-dependent floor across three variants.
- Characterized when thermal derating becomes the binding constraint, via a parametric cooling-degradation stress test: 0 binding events at nominal cooling, rising to 581 events (mean shortfall 26.9 kW) at 30% cooling effectiveness.
- Closed the loop between the DP policy and the thermal state through fixed-point iteration between `backward_process` and `thermal_model`, rather than a full 3D state reformulation, and made the tradeoff between the two approaches explicit.

## Motivation

Modern hybrid power units (ICE + MGU-K + Energy Store) require a real-time strategy that decides, at every instant, how much power comes from the combustion engine versus the electric motor-generator. This project builds a physics-based plant model of the powertrain from first principles, validates it against real qualifying telemetry, and uses it as a sandbox to compare energy management strategies of increasing sophistication, from simple heuristics to strategies that exploit full knowledge of the future velocity profile and of thermal constraints.

## Project structure

```
.
├── controller/
│   ├── 01_single_lap_DP.ipynb            # single-lap fuel-optimal DP (benchmark)
│   ├── 01b_single_lap_DP_vectorized.ipynb# vectorized rewrite, same result as 01
│   ├── 02_ECMS.ipynb                     # online-representative ECMS controller
│   ├── 03_multi_lap_DP.ipynb             # 5-lap DP + SoC repeatability constraint
│   ├── 04_multi_lap_DP_thermal.ipynb     # + thermal derating, fixed-point self-consistency
│   ├── Summary_of_Results.ipynb          # condensed results, start here
│   ├── rule_based_controller.py          # SoC-scheduled heuristic baseline
│   └── README.md                         # per-notebook status, results, limitations
├── plant/
│   ├── parameters.py                     # vehicle, ICE, MGU-K, battery parameters
│   ├── vehicle_dynamics.py               # longitudinal dynamics, power demand
│   ├── battery.py                        # equivalent-circuit battery + thermal model
│   └── powertrain.py                     # ICE (Willans line) + MGU-K + power-split bookkeeping
├── scripts/
│   ├── build_velocity_profile.py         # loads FastF1 telemetry, resamples, saves .npy
│   └── simulation.py                     # builds the gearbox power-demand array
├── data/
│   ├── qualifying_Canada/                # single-lap DP inputs/outputs (Canada 2026 quali)
│   ├── multi_lap_Canada/                 # synthetic 5-lap profile + battery temperature
│   └── results/                          # exported .npz feeding Summary_of_Results.ipynb
├── img/                                  # exported plots (qualifying and multi-lap profiles)
├── Compare_Controllers.ipynb             # head-to-head: rule-based vs ECMS vs DP
└── requirements.txt
```

## Plant model

The powertrain is modeled quasistatically (no fast electrical transients), following the modeling philosophy of Guzzella & Sciarretta, *Vehicle Propulsion Systems*, and validated against the convex formulation of Ebbesen et al. (2018), *Time-Optimal Control Strategies for a Hybrid Electric Race Car*, IEEE TCST.

- **Vehicle dynamics**: longitudinal force balance (aerodynamic drag, rolling resistance, inertia) converted to power demand at the gearbox, from a real velocity/acceleration profile extracted via [FastF1](https://github.com/theOehrly/Fast-F1) and Savitzky-Golay filtered to obtain a physically plausible acceleration signal.
- **ICE**: affine Willans-line model (`P_e = eta_ICE * P_fuel - P_ICE0`), calibrated so that the fuel-flow-achievable ICE power matches the 400 kW regulatory cap, with a regulatory fuel-flow limit.
- **MGU-K**: instantaneous power limits (`P_MGU_max`, `P_MGU_min`), a cumulative per-lap deploy energy budget (E_deploy_max, reset at every lap boundary), and a temperature-dependent derating factor on maximum discharge power.
- **Battery**: single-branch equivalent-circuit model (open-circuit voltage in series with internal resistance), solved via the closed-form quadratic solution for terminal voltage, with Coulomb counting for SoC integration. The OCV(SoC) curve is calibrated from a reference high-power Li-ion cell (Samsung INR21700-48X, per Rukavina et al., 2023) and rescaled to the target pack voltage window, since proprietary real pack data is not publicly available.
- **Thermal model**: lumped-capacitance battery temperature state driven by Joule heating (`I²R_int`) and convective cooling to a coolant at fixed inlet temperature, with a linear derating curve above a threshold temperature.
- **Power-split shortfall**: the plant never silently compensates a saturated component with another; any gap between requested and deliverable power is reported explicitly, to be handled by the controller.

## Control strategies

| Strategy | Knowledge of the future | Status |
|---|---|---|
| Rule-based (SoC-scheduled) | None | Implemented |
| ECMS | None (online-representative) | Implemented |
| Dynamic Programming, single lap | Full (benchmark / upper bound) | Implemented |
| Dynamic Programming, multi-lap + SoC soft constraint | Full | Implemented |
| Dynamic Programming + thermal derating | Full, thermally aware | Implemented |
| RL (SAC) | Learned policy, generalizable | Future work, deferred |

The rule-based controller schedules the MGU-K power fraction linearly on SoC during traction and maximizes regeneration during braking, subject to the MGU-K's physical and regulatory limits. It serves as a baseline against which the optimization-based strategies are benchmarked.

The multi-lap DP formulation deliberately keeps the backward pass simplified (no `R_int` losses in the stage cost) while the forward pass uses the full nonlinear plant, to quantify the model-plant mismatch as part of the validation. The soft constraint on end-of-lap SoC and the thermal derating bound are both surrogate corrections built on top of this same 2D grid, rather than promoting temperature to a third DP state, a design tradeoff made explicit in the notebooks and discussed further below.

## Installation

```bash
git clone https://github.com/tommasoandina1/Thermal-Aware-Energy-Management-System-for-a-Hybrid-Motorsport-Powertrain
cd Thermal-Aware-Energy-Management-System-for-a-Hybrid-Motorsport-Powertrain
pip install -r requirements.txt
```

## Usage

```bash
# 1. Build the velocity/acceleration profile from telemetry
python scripts/build_velocity_profile.py

# 2. Build the gearbox power-demand array
python scripts/simulation.py
```

For the full controller comparison, run the notebooks in `controller/` (see
`controller/README.md` for the recommended reading order), or open
`Compare_Controllers.ipynb` for the rule-based vs ECMS vs DP head-to-head.
Readers who only want the final results should start with
[`controller/Summary_of_Results.ipynb`](controller/Summary_of_Results.ipynb).

## Status and scope

This is an active portfolio project developed alongside a master's thesis in Mechatronics Engineering (Politecnico di Torino). The plant model, rule-based, ECMS and DP controllers (single and multi-lap, with SoC repeatability and thermal derating) are complete; RL is deferred as future work pending time availability before the September 2026 deadline.

Known simplifications, made explicit rather than hidden:

- No turbocharger sub-model, no engine-speed dependence (per the Willans-line simplification validated in Ebbesen et al.).
- No hydraulic-brake energy dissipation model (reported as power shortfall instead).
- `P_MGU-K` is controlled in electrical terms in the DP/ECMS formulation (`eta_MGU = 1` approximation in the power balance), while the forward pass uses the real plant to quantify the resulting mismatch.
- The thermal derating bound is enforced open-loop and corrected via fixed-point iteration between the DP policy and the thermal model, not via a full 3D state reformulation; this is a deliberate cost/accuracy tradeoff, discussed in `controller/04_multi_lap_DP_thermal.ipynb`.
- `R_int` is treated as constant (not temperature-dependent); the OCV curve and thermal parameters are estimated from published reference data, not calibrated against a real pack, since proprietary data is unavailable.

## References

- Guzzella, L., Sciarretta, A. *Vehicle Propulsion Systems*, 3rd ed., Springer, 2013.
- Ebbesen, S., Salazar, M., Elbert, P., Bussi, C., Onder, C.H. "Time-Optimal Control Strategies for a Hybrid Electric Race Car." *IEEE Transactions on Control Systems Technology*, 26(1), 2018.
- Rukavina, F., Leko, D., Matijašić, M., Bralić, I., Ugalde, J.M., Vašak, M. "Identification of equivalent circuit model parameters for a Li-ion battery cell." *Proc. 2023 IEEE 11th International Conference on Systems and Control*.

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.
