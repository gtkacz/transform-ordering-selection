"""Generate all three figures for the IEEE paper from experimental results."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# IEEE single-column width: ~3.5in, double-column: ~7.16in
# We target single-column figures at 300 DPI
IEEE_COL_WIDTH = 3.5
IEEE_FONT_SIZE = 8
IEEE_TICK_SIZE = 7

TRANSFORM_ABBREV = {
    "EqualizationTransform": "E",
    "NormalizeTransform": "N",
    "DenoiseTransform": "D",
    "ColorSpaceTransform": "CS",
}

TRANSFORM_FULL = {
    "EqualizationTransform": "Equalization",
    "NormalizeTransform": "Normalization",
    "DenoiseTransform": "Denoising",
    "ColorSpaceTransform": "Color Space",
}


def setup_ieee_style():
    """Configure matplotlib for IEEE-quality output."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": IEEE_FONT_SIZE,
            "axes.titlesize": IEEE_FONT_SIZE,
            "axes.labelsize": IEEE_FONT_SIZE,
            "xtick.labelsize": IEEE_TICK_SIZE,
            "ytick.labelsize": IEEE_TICK_SIZE,
            "legend.fontsize": IEEE_TICK_SIZE,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "patch.linewidth": 0.5,
            "axes.grid": False,
            "text.usetex": False,
        }
    )


def load_data(results_path: str) -> list[dict]:
    with open(results_path) as f:
        data = json.load(f)

    base = [d for d in data if d["confidence_level"] == "base"]
    for d in base:
        d["length"] = len(d["transforms"])
    return base


def fig1_length_effect(base: list[dict], outdir: Path):
    """Box plot of alpha by pipeline length with statistical annotations."""
    non_baseline = [d for d in base if d["length"] > 0]

    by_length: dict[int, list[float]] = defaultdict(list)
    for d in non_baseline:
        by_length[d["length"]].append(d["alpha"] * 100)

    lengths = [1, 2, 3, 4]
    box_data = [by_length[L] for L in lengths]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.4))

    bp = ax.boxplot(
        box_data,
        positions=lengths,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "gray", "markeredgecolor": "gray", "alpha": 0.7},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )

    grays = ["#D9D9D9", "#BFBFBF", "#A6A6A6", "#808080"]
    for patch, color in zip(bp["boxes"], grays):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    for i, L in enumerate(lengths):
        vals = box_data[i]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), L) + jitter,
            vals,
            s=8,
            c="black",
            alpha=0.5,
            zorder=5,
            edgecolors="none",
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.6)

    for i, L in enumerate(lengths):
        vals = box_data[i]
        n_pos = sum(1 for v in vals if v > 0)
        pct = n_pos / len(vals) * 100
        y_top = max(vals) + 0.3
        ax.text(
            L,
            y_top,
            f"{pct:.0f}%+",
            ha="center",
            va="bottom",
            fontsize=6,
            fontstyle="italic",
        )

    ax.text(
        0.97,
        0.03,
        "$r = -0.421$, $p = 0.0002$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray", "linewidth": 0.5},
    )

    ax.set_xlabel("Pipeline length (number of transforms)")
    ax.set_ylabel("Accuracy gain $\\alpha$ (pp)")
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{L}\n($n$={len(by_length[L])})" for L in lengths])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "fig_length_effect.pdf")
    fig.savefig(outdir / "fig_length_effect.png")
    plt.close(fig)
    print("  [OK] fig_length_effect.pdf")


def fig2_variance_decomposition(base: list[dict], outdir: Path):
    """Stacked bar chart of selection vs ordering variance at each pipeline length."""
    results = {}

    for L in [2, 3]:
        entries = [d for d in base if d["length"] == L]
        by_set: dict[tuple, list[float]] = defaultdict(list)
        for d in entries:
            key = tuple(sorted(d["transforms"]))
            by_set[key].append(d["alpha"])

        all_alphas = [d["alpha"] for d in entries]
        grand_mean = np.mean(all_alphas)

        ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in by_set.values())
        ss_within = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in by_set.values())
        ss_total = ss_between + ss_within

        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        results[L] = {"selection": eta_sq, "ordering": 1 - eta_sq}

    results[4] = {"selection": 0.0, "ordering": 1.0}

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.2))

    bar_lengths = [2, 3, 4]
    x = np.arange(len(bar_lengths))
    width = 0.55

    sel_vals = [results[L]["selection"] * 100 for L in bar_lengths]
    ord_vals = [results[L]["ordering"] * 100 for L in bar_lengths]

    bars_sel = ax.bar(x, sel_vals, width, label="Selection (between-set)", color="#D9D9D9", edgecolor="black", linewidth=0.5)
    bars_ord = ax.bar(x, ord_vals, width, bottom=sel_vals, label="Ordering (within-set)", color="#606060", edgecolor="black", linewidth=0.5)

    for i, L in enumerate(bar_lengths):
        s = results[L]["selection"] * 100
        o = results[L]["ordering"] * 100
        if s > 8:
            ax.text(x[i], s / 2, f"{s:.1f}%", ha="center", va="center", fontsize=6, color="black")
        if o > 8:
            ax.text(x[i], s + o / 2, f"{o:.1f}%", ha="center", va="center", fontsize=6, color="white")

    n_per_length = {2: 12, 3: 24, 4: 24}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n($n$={n_per_length[L]})" for L in bar_lengths])
    ax.set_xlabel("Pipeline length")
    ax.set_ylabel("Proportion of variance (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=6.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the crossover
    ax.annotate(
        "crossover",
        xy=(0.5, 55),
        xytext=(1.4, 20),
        fontsize=6,
        fontstyle="italic",
        arrowprops={"arrowstyle": "->", "linewidth": 0.5, "color": "gray"},
        color="gray",
    )

    fig.tight_layout()
    fig.savefig(outdir / "fig_variance_decomp.pdf")
    fig.savefig(outdir / "fig_variance_decomp.png")
    plt.close(fig)
    print("  [OK] fig_variance_decomp.pdf")


def fig3_positional_preferences(base: list[dict], outdir: Path):
    """Line plot of mean alpha by ordinal position for each transform."""
    non_baseline = [d for d in base if d["length"] > 0]

    transform_order = [
        "EqualizationTransform",
        "NormalizeTransform",
        "DenoiseTransform",
        "ColorSpaceTransform",
    ]

    markers = ["s", "^", "o", "D"]
    linestyles = ["-", "-", "--", "--"]
    colors = ["black", "#606060", "#909090", "#B0B0B0"]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.4))

    for idx, tname in enumerate(transform_order):
        pos_alphas: dict[int, list[float]] = defaultdict(list)
        for d in non_baseline:
            if tname in d["transforms"]:
                pos = d["transforms"].index(tname) + 1
                pos_alphas[pos].append(d["alpha"] * 100)

        positions = sorted(pos_alphas.keys())
        means = [np.mean(pos_alphas[p]) for p in positions]
        sems = [np.std(pos_alphas[p]) / np.sqrt(len(pos_alphas[p])) for p in positions]

        label = f"{TRANSFORM_ABBREV[tname]} ({TRANSFORM_FULL[tname]})"
        ax.errorbar(
            positions,
            means,
            yerr=sems,
            marker=markers[idx],
            linestyle=linestyles[idx],
            color=colors[idx],
            label=label,
            capsize=2,
            capthick=0.5,
            markerfacecolor=colors[idx] if idx < 2 else "white",
            markeredgecolor=colors[idx],
            markeredgewidth=0.7,
        )

    ax.axhline(y=0, color="black", linestyle=":", linewidth=0.4, alpha=0.5)

    ax.set_xlabel("Ordinal position in pipeline")
    ax.set_ylabel("Mean accuracy gain $\\alpha$ (pp)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1\n(first)", "2", "3", "4\n(last)"])

    ax.legend(loc="lower left", fontsize=6, frameon=True, fancybox=False, edgecolor="gray", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the gradients for E and N
    ax.annotate(
        "$+0.60$ pp/pos",
        xy=(1.2, -0.55),
        fontsize=5.5,
        fontstyle="italic",
        color="black",
    )
    ax.annotate(
        "$+0.32$ pp/pos",
        xy=(3.05, -0.7),
        fontsize=5.5,
        fontstyle="italic",
        color="#606060",
    )

    fig.tight_layout()
    fig.savefig(outdir / "fig_positional.pdf")
    fig.savefig(outdir / "fig_positional.png")
    plt.close(fig)
    print("  [OK] fig_positional.pdf")


def fig4_gradcam_comparison(outdir: Path):
    """2x2 panel: same image under mirror-ordering pipelines E->N vs N->E."""
    from PIL import Image

    output_base = Path(__file__).resolve().parent.parent.parent / "src" / "output" / "base"

    panels = {
        (0, 0): output_base / "EqualizationTransform -> NormalizeTransform" / "gradcam_pos_img_00002.png",
        (0, 1): output_base / "EqualizationTransform -> NormalizeTransform" / "gradcam_neg_img_00002.png",
        (1, 0): output_base / "NormalizeTransform -> EqualizationTransform" / "gradcam_pos_img_00002.png",
        (1, 1): output_base / "NormalizeTransform -> EqualizationTransform" / "gradcam_neg_img_00002.png",
    }

    for pos, path in panels.items():
        if not path.exists():
            print(f"  [SKIP] fig_gradcam.pdf — missing {path.name}", file=sys.stderr)
            return

    row_labels = [
        r"E $\to$ N ($\alpha = +0.58$ pp)",
        r"N $\to$ E ($\alpha = -1.33$ pp)",
    ]
    col_labels = ["Positive-class activation", "Negative-class activation"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH * 0.95),
        constrained_layout=True,
    )

    for (r, c), path in panels.items():
        img = np.array(Image.open(path))
        axes[r, c].imshow(img, interpolation="lanczos")
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        for spine in axes[r, c].spines.values():
            spine.set_linewidth(0.4)

    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=IEEE_TICK_SIZE, pad=4)

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=IEEE_TICK_SIZE, labelpad=4)

    panel_ids = [["(a)", "(b)"], ["(c)", "(d)"]]
    for r in range(2):
        for c in range(2):
            axes[r, c].text(
                0.03, 0.95, panel_ids[r][c],
                transform=axes[r, c].transAxes,
                fontsize=IEEE_TICK_SIZE,
                fontweight="bold",
                va="top",
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.5, "linewidth": 0},
            )

    fig.savefig(outdir / "fig_gradcam.pdf")
    fig.savefig(outdir / "fig_gradcam.png")
    plt.close(fig)
    print("  [OK] fig_gradcam.pdf")


def main():
    results_path = Path(__file__).resolve().parent.parent.parent / "src" / "output" / "results_matrix.json"
    outdir = Path(__file__).resolve().parent

    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {results_path}")
    base = load_data(str(results_path))
    print(f"  {len(base)} base-regime entries loaded")

    setup_ieee_style()

    print("Generating figures...")
    fig1_length_effect(base, outdir)
    fig2_variance_decomposition(base, outdir)
    fig3_positional_preferences(base, outdir)
    fig4_gradcam_comparison(outdir)

    print("All figures generated.")


if __name__ == "__main__":
    main()
