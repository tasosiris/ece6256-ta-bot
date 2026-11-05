#!/usr/bin/env python3
"""Plot ablation results as hit-rate bars for the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
SERIES = {
    "rewrite_off": ("Rewriting off", "#2a78d6", ""),
    "rewrite_on": ("Rewriting on", "#eb6834", "///"),  # hatch keeps it readable in grayscale
}
KIND_LABELS = {"followup": "Follow-ups", "control": "Controls (topic switches)"}
BAR_WIDTH = 0.34
BAR_GAP = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HERE / "results.json"))
    parser.add_argument("--out", default=str(HERE / "ablation.png"))
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    kinds = [kind for kind in KIND_LABELS if kind in results["summary"]]

    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=300, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for offset, (condition, (label, color, hatch)) in zip((-1, 1), SERIES.items()):
        xs = [i + offset * (BAR_WIDTH + BAR_GAP) / 2 for i in range(len(kinds))]
        stats = [results["summary"][kind][condition] for kind in kinds]
        rates = [s["hit_rate"] for s in stats]
        errors = [
            [r - s["ci_low"] for r, s in zip(rates, stats)],
            [s["ci_high"] - r for r, s in zip(rates, stats)],
        ]
        ax.bar(
            xs, rates, BAR_WIDTH, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0.5,
            yerr=errors, ecolor=INK_SECONDARY, error_kw={"elinewidth": 1, "capsize": 3},
            label=label, zorder=2,
        )
        for x, rate in zip(xs, rates):
            ax.text(x, 0.02, f"{rate:.0%}", ha="center", va="bottom", fontsize=8,
                    color=INK, zorder=3,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})

    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(
        [f'{KIND_LABELS[k]}\nn = {results["summary"][k]["n"]}' for k in kinds],
        fontsize=8, color=INK_SECONDARY,
    )
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Gold paper cited", fontsize=8, color=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(
        frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        labelcolor=INK_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
