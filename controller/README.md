# `controller/` — control strategies

The five notebooks build up in order, each one adding a single ingredient to the
previous formulation. `Summary_of_Results.ipynb` reads their saved outputs and
reproduces the headline figures without recomputing the backward passes.

**If you only read one:** [`Summary_of_Results.ipynb`](Summary_of_Results.ipynb).

## Reading order

| notebook | what it adds | key figure |
|---|---|---|
| [`01_single_lap_DP.ipynb`](01_single_lap_DP.ipynb) | Backward DP on a 2D grid (SoC, deploy energy) over one qualifying lap, with full knowledge of the velocity profile. The offline benchmark everything else is measured against. | 805.48 g fuel |
| [`01b_single_lap_DP_vectorized.ipynb`](01b_single_lap_DP_vectorized.ipynb) | The same recursion rewritten with grid-wide NumPy operations instead of explicit state loops. Implementation only — it verifies numerical equivalence with `01`, it does not restate the physics. | same result as `01` |
| [`02_ECMS.ipynb`](02_ECMS.ipynb) | Instantaneous minimisation of an equivalent fuel cost, no recursion. The equivalence factor is anchored at 1/η_ICE = 2.008 and corrected by proportional feedback on a SoC reference. | 820.09 g fuel |
| [`03_multi_lap_DP.ipynb`](03_multi_lap_DP.ipynb) | Five laps. Shows the fuel-optimal baseline draining the pack within two laps, then compares a fixed and a lap-dependent soft floor on end-of-lap SoC. | unmet demand 1.79 → 0.90 MJ, +3.5% fuel |
| [`04_multi_lap_DP_thermal.ipynb`](04_multi_lap_DP_thermal.ipynb) | A temperature-dependent ceiling on MGU-K deploy power, plus a parametric cooling stress test and the fixed-point iteration that makes policy and thermal state agree. | 261 binding instants at nominal cooling, 1504 at 30% |

`rule_based_controller.py` is the heuristic baseline used in
[`../Compare_Controllers.ipynb`](../Compare_Controllers.ipynb): the MGU-K power
fraction is scheduled linearly on SoC during traction, regeneration is maximised
during braking, both subject to the MGU-K envelope and the per-lap deploy budget.

## Why `01` and `01b` both exist

`01` is the readable reference implementation — explicit loops, one state
transition per line, the formulation visible in the code. `01b` is the version
actually used when a backward pass has to be rerun, roughly two orders of
magnitude faster. They are kept side by side because the vectorised recursion is
hard to read against the Bellman equation it implements, and the pair documents
the translation. `01b` is the one to copy if you are extending the solver.

## Conventions

Notebooks in this directory run from the repository root, not from
`controller/`. Each notebook states its inputs, outputs and plant dependencies
in its opening cell.
