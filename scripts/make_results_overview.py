"""
Regenerate the summary figure shown at the top of the README.

Reads the exported result archives under data/results/ and produces
img/results_overview.png. Nothing is recomputed here: the figure is a view of
results produced by the notebooks, so it cannot silently disagree with them.

    python scripts/make_results_overview.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import RESULTS, IMG, ensure_dirs  # noqa: E402

INK = "#1b1b1b"
GRID = "#d8d8d8"
C_DP = "#0b6e4f"
C_ECMS = "#c1666b"
C_RB = "#7b8794"
C_WARN = "#d1495b"


def style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8, length=3)


def main():
    ensure_dirs()
    single = np.load(RESULTS / "single_lap_comparison.npz")
    multi = np.load(RESULTS / "multi_lap_comparison.npz")
    therm = np.load(RESULTS / "thermal_analysis.npz")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    fig.patch.set_facecolor("white")

    # --- panel 1: single lap, charge trajectories -------------------------
    ax = axes[0]
    t = single["t"]
    fuel = {k: single[f"J_{k}"].sum() * 1e3 for k in ("dp", "ecms", "rb")}
    ax.plot(t, single["SoC_rb"], color=C_RB, lw=1.4,
            label=f"rule-based  {fuel['rb']:.0f} g")
    ax.plot(t, single["SoC_ecms"], color=C_ECMS, lw=1.6, ls="--",
            label=f"ECMS  {fuel['ecms']:.0f} g")
    ax.plot(t, single["SoC_dp"], color=C_DP, lw=1.8,
            label=f"DP (optimal)  {fuel['dp']:.0f} g")
    ax.set_title("Knowing the lap ahead is worth 1.8% fuel",
                 fontsize=10, color=INK, loc="left", pad=9)
    ax.set_xlabel("time [s]", fontsize=8)
    ax.set_ylabel("state of charge [-]", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    style(ax)

    # --- panel 2: five laps, repeatability --------------------------------
    ax = axes[1]
    tm, lap = multi["t"], multi["N_lap"]
    lap_ids = np.unique(lap)

    def per_lap_min(soc):
        return np.array([soc[lap == l].min() for l in lap_ids])

    for key, colour, lab in (("baseline", C_WARN, "fuel-optimal"),
                             ("fixed", C_DP, "with end-of-lap charge floor")):
        soc = multi[f"SoC_{key}"]
        ax.plot(np.arange(1, len(lap_ids) + 1), per_lap_min(soc), "o-",
                color=colour, lw=2.2, ms=7, label=lab)

    ax.axhline(0.2, color=INK, lw=1.0, ls=":", alpha=0.7)
    ax.text(len(lap_ids) + 0.05, 0.207, "charge floor", ha="right",
            fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(1, len(lap_ids) + 1))
    ax.set_xlim(0.6, len(lap_ids) + 0.4)
    ax.set_ylim(0.13, 0.88)
    ax.set_title("The fuel-optimal strategy is not raceable",
                 fontsize=10, color=INK, loc="left", pad=9)
    ax.set_xlabel("lap", fontsize=8)
    ax.set_ylabel("lowest state of charge in the lap [-]", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=1)
    style(ax)

    # --- panel 3: thermal derating ----------------------------------------
    ax = axes[2]
    cfs = therm["cfs"]
    counts = therm["binding_counts"]
    x = np.arange(len(cfs))
    bars = ax.bar(x, counts, width=0.6, color=C_WARN, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0%}" for c in cfs], fontsize=8)
    for xi, c in zip(x, counts):
        ax.text(xi, c + 25, f"{c}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title("Thermal derating binds before charge runs out",
                 fontsize=10, color=INK, loc="left", pad=9)
    ax.set_xlabel("cooling effectiveness", fontsize=8)
    ax.set_ylabel("stages with derating active\n(out of 3886)", fontsize=8)
    style(ax)

    sf_b, sf_a = float(therm["sf_before"]), float(therm["sf_after"])
    fig.text(0.995, 0.015,
             f"closing the policy-thermal loop: unmet demand "
             f"{sf_b:.2f} → {sf_a:.2f} MJ",
             ha="right", fontsize=8, color=C_RB, style="italic")

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = IMG / "results_overview.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  single lap fuel [g]: " +
          ", ".join(f"{k}={v:.2f}" for k, v in fuel.items()))
    print(f"  binding counts: {dict(zip(cfs.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
