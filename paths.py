"""
Single source of truth for every path in the project.

Motivation: notebooks run with their own directory as the working directory,
scripts run from the repository root, and the Docker image mounts the project
at /app. A relative literal such as 'data/results/thermal_analysis.npz' is
therefore correct in one of those three contexts and silently wrong in the
other two - it either raises FileNotFoundError or, worse, writes a second copy
of the results somewhere nobody looks.

Every path below is derived from the location of this file, so it resolves to
the same absolute location regardless of where Python was started.

Usage
-----
    from paths import DATA_QUALI, RESULTS, ensure_dirs

    ensure_dirs()
    np.save(DATA_QUALI / "SOC_DP_Canada_qualifying.npy", SoC)
    np.savez(RESULTS / "thermal_analysis.npz", **payload)

In a notebook under controller/, add the repository root to the path first:

    import sys; sys.path.insert(0, "..")
    from paths import RESULTS
"""

from pathlib import Path
import os

# --- roots -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

PLANT = ROOT / "plant"
CONTROLLER = ROOT / "controller"
SCRIPTS = ROOT / "scripts"
RL = ROOT / "rl"
TESTS = ROOT / "tests"

# --- data ------------------------------------------------------------------

DATA = ROOT / "data"
DATA_QUALI = DATA / "qualifying_Canada"      # single-lap DP inputs and outputs
DATA_MULTI = DATA / "multi_lap_Canada"       # 5-lap profile and battery temperature
RESULTS = DATA / "results"                   # .npz consumed by Summary_of_Results

# --- outputs ---------------------------------------------------------------

IMG = ROOT / "img"                           # exported figures
RUNS = ROOT / "runs"                         # RL training runs (checkpoints, logs)

# --- named files -----------------------------------------------------------
# Referenced from more than one place, so they get a name instead of a literal.

VELOCITY_PROFILE_QUALI = DATA_QUALI / "Canada_qualifying.npy"
VELOCITY_PROFILE_MULTI = DATA_MULTI / "Canada_5laps.npy"

SINGLE_LAP_COMPARISON = RESULTS / "single_lap_comparison.npz"
MULTI_LAP_COMPARISON = RESULTS / "multi_lap_comparison.npz"
THERMAL_ANALYSIS = RESULTS / "thermal_analysis.npz"

# --- FastF1 cache ----------------------------------------------------------
# Outside the repository by default. Keeping it inside means every telemetry
# download lands in the working tree and, sooner or later, in git history -
# which is how a 61 MB cache directory got committed in the first place.
# The Docker image sets FASTF1_CACHE=/cache on a named volume.

FASTF1_CACHE = Path(os.environ.get(
    "FASTF1_CACHE",
    Path.home() / ".cache" / "fastf1_thermal_aware_ems"))


def ensure_dirs():
    """Create the directories that are written to. Safe to call repeatedly."""
    for d in (DATA_QUALI, DATA_MULTI, RESULTS, IMG, RUNS, FASTF1_CACHE):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    for name, value in sorted(globals().items()):
        if name.isupper() and isinstance(value, Path):
            mark = "ok " if value.exists() else "-- "
            print(f"{mark}{name:26s} {value}")
