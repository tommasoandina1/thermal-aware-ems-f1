# Thermal-Aware Energy Management System for a Hybrid Motorsport Powertrain

A quasistatic simulation and control framework for the energy management strategy (EMS) of a hybrid motorsport powertrain (F1 2026 / LMDh-inspired), built from real qualifying telemetry. The project benchmarks a rule-based baseline against Equivalent Consumption Minimization Strategy (ECMS), Dynamic Programming (DP), and Reinforcement Learning (SAC) controllers for lap-time and energy-optimal power-split decisions.

## Motivation

Modern hybrid power units (ICE + MGU-K + Energy Store) require a real-time strategy that decides, at every instant, how much power comes from the combustion engine versus the electric motor-generator. This project builds a physics-based plant model of the powertrain from first principles, validates it against a real qualifying lap, and uses it as a sandbox to compare energy management strategies of increasing sophistication — from simple heuristics to strategies that exploit full knowledge of the future velocity profile.

## Project structure

```
.
├── data/
│   └── spanish_qualifying.npy       # Resampled velocity/acceleration profile (fastf1)
├── plant/
│   ├── parameters.py                # Vehicle, ICE, MGU-K and battery parameters
│   ├── vehicle_dynamics.py          # Longitudinal dynamics: aero drag, rolling resistance, power demand
│   ├── battery.py                   # Quasistatic equivalent-circuit battery model (OCV + internal resistance)
│   └── powertrain.py                # ICE (Willans line) + MGU-K + power-split bookkeeping
├── control/
│   ├── rule_based.py                # SoC-scheduled heuristic power-split baseline
│   ├── ecms.py                      # Equivalent Consumption Minimization Strategy (planned)
│   └── rl_agent/                    # Gymnasium environment + SAC agent (planned)
├── scripts/
│   ├── build_velocity_profile.py    # Loads fastf1 telemetry, resamples, filters, saves .npy
│   └── run_simulation.py            # End-to-end lap simulation and plotting
└── requirements.txt
```

## Plant model

The powertrain is modeled quasistatically (no fast electrical/thermal transients), following the modeling philosophy of Guzzella & Sciarretta, *Vehicle Propulsion Systems*, and validated against the convex formulation of Ebbesen et al. (2018), *Time-Optimal Control Strategies for a Hybrid Electric Race Car*, IEEE TCST.

- **Vehicle dynamics**: longitudinal force balance (aerodynamic drag, rolling resistance, inertia) converted to power demand at the gearbox, from a real velocity/acceleration profile extracted via [FastF1](https://github.com/theOehrly/Fast-F1) and Savitzky-Golay filtered to obtain a physically plausible acceleration signal.
- **ICE**: affine Willans-line model (`P_e = eta_ICE * P_fuel - P_ICE0`), with a regulatory fuel-flow limit.
- **MGU-K**: direction-dependent efficiency (motoring vs regeneration), instantaneous power limits, and a cumulative per-lap deploy energy budget (ES2K-style constraint), enforced as permanent saturation once exhausted.
- **Battery**: single-branch equivalent-circuit model (open-circuit voltage in series with internal resistance), solved via the closed-form quadratic solution for terminal voltage, with Coulomb counting for SoC integration. The OCV(SoC) curve is calibrated from a reference high-power Li-ion cell (Samsung INR21700-48X, per Rukavina et al., 2023) and rescaled to the target pack voltage window, since proprietary real pack data is not publicly available.
- **Power-split shortfall**: the plant never silently compensates a saturated component with another; any gap between requested and deliverable power is reported explicitly, to be handled by the controller.

## Control strategies

|
 Strategy 
|
 Knowledge of the future 
|
 Status 
---
|
---
|
---
|
|
 Rule-based (SoC-scheduled) 
|
 None 
|
 Implemented 
|
|
 ECMS 
|
 None (online-representative) 
|
 Planned 
|
|
 Dynamic Programming 
|
 Full (benchmark / upper bound) 
|
 Planned 
|
|
 RL (SAC) 
|
 Learned policy, generalizable 
|
 Planned 
|

The rule-based controller schedules the MGU-K power fraction linearly on SoC during traction and maximizes regeneration during braking, subject to the MGU-K's physical and regulatory limits. It serves as a baseline against which the optimization-based strategies are benchmarked.

## Installation

```bash
git clone https://github.com//thermal-aware-ems.git
cd thermal-aware-ems
pip install -r requirements.txt
```

## Usage

```bash
# 1. Build the velocity/acceleration profile from telemetry
python scripts/build_velocity_profile.py

# 2. Run the end-to-end lap simulation
python scripts/run_simulation.py
```

## Status and scope

This is an active portfolio project developed alongside a master's thesis in Mechatronics Engineering (Politecnico di Torino / University of Illinois Chicago). The current focus is the plant model and rule-based baseline; thermal modeling, ECMS, DP, and RL components are in progress. Known simplifications (documented in the code) include: no turbocharger sub-model, no engine-speed dependence (per the Willans-line simplification validated in Ebbesen et al.), and no hydraulic-brake energy dissipation model (reported as power shortfall instead).

## References

- Guzzella, L., Sciarretta, A. *Vehicle Propulsion Systems*, 3rd ed., Springer, 2013.
- Ebbesen, S., Salazar, M., Elbert, P., Bussi, C., Onder, C.H. "Time-Optimal Control Strategies for a Hybrid Electric Race Car." *IEEE Transactions on Control Systems Technology*, 26(1), 2018.
- Rukavina, F., Leko, D., Matijašić, M., Bralić, I., Ugalde, J.M., Vašak, M. "Identification of equivalent circuit model parameters for a Li-ion battery cell." *Proc. 2023 IEEE 11th International Conference on Systems and Control*.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
