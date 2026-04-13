"""Generate all figures for the IEEE TIP paper from multi-seed analysis results."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ELSEVIER_COL_WIDTH = 3.5
ELSEVIER_FONT_SIZE = 9
ELSEVIER_TICK_SIZE = 8

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


def setup_elsevier_style():
	"""Configure matplotlib for Elsevier-quality output."""
	plt.rcParams.update({
		"font.family": "serif",
		"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
		"font.size": ELSEVIER_FONT_SIZE,
		"axes.titlesize": ELSEVIER_FONT_SIZE,
		"axes.labelsize": ELSEVIER_FONT_SIZE,
		"xtick.labelsize": ELSEVIER_TICK_SIZE,
		"ytick.labelsize": ELSEVIER_TICK_SIZE,
		"legend.fontsize": ELSEVIER_TICK_SIZE,
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
	})


def load_analysis(analysis_path: Path) -> dict:
	"""Load the aggregated multi-seed analysis JSON."""
	with analysis_path.open() as f:
		return json.load(f)


def fig1_length_effect(analysis: dict, outdir: Path):
	"""Box plot of alpha by pipeline length with multi-seed mean alpha values."""
	base = analysis["base"]
	pipes = [p for p in base["aggregated_pipelines"] if p["pipeline_length"] > 0]

	by_length: dict[int, list[float]] = defaultdict(list)
	for p in pipes:
		by_length[p["pipeline_length"]].append(p["mean_alpha"] * 100)

	lengths = [1, 2, 3, 4]
	box_data = [by_length[L] for L in lengths]

	fig, ax = plt.subplots(figsize=(ELSEVIER_COL_WIDTH, 2.4))

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

	r = base["statistical_tests"]["length_degradation"]["correlation"]["observed_statistic"]
	corrected = next(
		c for c in base["statistical_tests"]["corrected_p_values"] if c["test_name"] == "length_correlation"
	)
	p_corr = corrected["corrected_p"]
	ax.text(
		0.97,
		0.03,
		f"$r = {r:+.2f}$, $p_{{\\mathrm{{Holm}}}} = {p_corr:.4f}$",
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


def fig2_variance_decomposition(analysis: dict, outdir: Path):
	"""Stacked bar chart of selection vs ordering variance at each pipeline length."""
	vd = analysis["base"]["statistical_tests"]["variance_decomposition"]

	results = {}
	for L_key in ("2", "3"):
		eta_sq = vd[L_key]["anova"]["eta_squared"]
		results[int(L_key)] = {"selection": eta_sq, "ordering": 1 - eta_sq}
	results[4] = {"selection": 0.0, "ordering": 1.0}

	fig, ax = plt.subplots(figsize=(ELSEVIER_COL_WIDTH, 2.2))

	bar_lengths = [2, 3, 4]
	x = np.arange(len(bar_lengths))
	width = 0.55

	sel_vals = [results[L]["selection"] * 100 for L in bar_lengths]
	ord_vals = [results[L]["ordering"] * 100 for L in bar_lengths]

	ax.bar(
		x, sel_vals, width, label="Selection (between-set)", color="#D9D9D9", edgecolor="black", linewidth=0.5,
	)
	ax.bar(
		x,
		ord_vals,
		width,
		bottom=sel_vals,
		label="Ordering (within-set)",
		color="#606060",
		edgecolor="black",
		linewidth=0.5,
	)

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


def fig3_positional_preferences(analysis: dict, outdir: Path):
	"""Line plot of mean alpha by ordinal position for each transform with SEMs."""
	pipes = [p for p in analysis["base"]["aggregated_pipelines"] if p["pipeline_length"] > 0]

	transform_order = [
		"EqualizationTransform",
		"NormalizeTransform",
		"DenoiseTransform",
		"ColorSpaceTransform",
	]

	markers = ["s", "^", "o", "D"]
	linestyles = ["-", "-", "--", "--"]
	colors = ["black", "#606060", "#909090", "#B0B0B0"]

	fig, ax = plt.subplots(figsize=(ELSEVIER_COL_WIDTH, 2.4))

	gradients = {}
	for idx, tname in enumerate(transform_order):
		pos_alphas: dict[int, list[float]] = defaultdict(list)
		for p in pipes:
			if tname in p["transforms"]:
				pos = p["transforms"].index(tname) + 1
				pos_alphas[pos].append(p["mean_alpha"] * 100)

		positions = sorted(pos_alphas.keys())
		means = [float(np.mean(pos_alphas[pos])) for pos in positions]
		sems = [float(np.std(pos_alphas[pos], ddof=1) / np.sqrt(len(pos_alphas[pos]))) for pos in positions]

		slope = float(np.polyfit(positions, means, 1)[0])
		gradients[tname] = slope

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

	e_slope = gradients["EqualizationTransform"]
	n_slope = gradients["NormalizeTransform"]
	ax.annotate(
		f"${e_slope:+.2f}$ pp/pos",
		xy=(1.2, -1.0),
		fontsize=5.5,
		fontstyle="italic",
		color="black",
	)
	ax.annotate(
		f"${n_slope:+.2f}$ pp/pos",
		xy=(3.05, -1.3),
		fontsize=5.5,
		fontstyle="italic",
		color="#606060",
	)

	fig.tight_layout()
	fig.savefig(outdir / "fig_positional.pdf")
	fig.savefig(outdir / "fig_positional.png")
	plt.close(fig)
	print("  [OK] fig_positional.pdf")


def _resolve_gradcam_panels() -> dict | None:
	"""Locate Grad-CAM panel PNGs for the E->N vs N->E comparison, preferring seed_42."""
	root = Path(__file__).resolve().parent.parent.parent / "src" / "output"
	search_roots = [root / "seed_42" / "base", root / "base"]
	for base_dir in search_roots:
		panels = {
			(0, 0): base_dir / "EqualizationTransform -> NormalizeTransform" / "gradcam_pos_img_00002.png",
			(0, 1): base_dir / "EqualizationTransform -> NormalizeTransform" / "gradcam_neg_img_00002.png",
			(1, 0): base_dir / "NormalizeTransform -> EqualizationTransform" / "gradcam_pos_img_00002.png",
			(1, 1): base_dir / "NormalizeTransform -> EqualizationTransform" / "gradcam_neg_img_00002.png",
		}
		if all(p.exists() for p in panels.values()):
			return panels
	return None


def fig4_gradcam_comparison(outdir: Path):
	"""2x2 panel: same image under mirror-ordering pipelines E->N vs N->E."""
	from PIL import Image

	panels = _resolve_gradcam_panels()
	if panels is None:
		print("  [SKIP] fig_gradcam.pdf — panel PNGs not found under src/output/**/base/", file=sys.stderr)
		return

	row_labels = [
		r"E $\to$ N",
		r"N $\to$ E",
	]
	col_labels = ["Positive-class activation", "Negative-class activation"]

	fig, axes = plt.subplots(
		2,
		2,
		figsize=(ELSEVIER_COL_WIDTH, ELSEVIER_COL_WIDTH * 0.95),
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
		axes[0, c].set_title(label, fontsize=ELSEVIER_TICK_SIZE, pad=4)

	for r, label in enumerate(row_labels):
		axes[r, 0].set_ylabel(label, fontsize=ELSEVIER_TICK_SIZE, labelpad=4)

	panel_ids = [["(a)", "(b)"], ["(c)", "(d)"]]
	for r in range(2):
		for c in range(2):
			axes[r, c].text(
				0.03,
				0.95,
				panel_ids[r][c],
				transform=axes[r, c].transAxes,
				fontsize=ELSEVIER_TICK_SIZE,
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
	analysis_path = Path(__file__).resolve().parent.parent.parent / "src" / "output" / "analysis.json"
	outdir = Path(__file__).resolve().parent

	if not analysis_path.exists():
		print(f"ERROR: {analysis_path} not found", file=sys.stderr)
		sys.exit(1)

	print(f"Loading multi-seed analysis from {analysis_path}")
	analysis = load_analysis(analysis_path)
	n_pipes = len(analysis["base"]["aggregated_pipelines"])
	n_seeds = analysis["base"]["metadata"]["n_seeds"]
	print(f"  {n_pipes} aggregated pipelines over {n_seeds} seeds")

	setup_elsevier_style()

	print("Generating figures...")
	fig1_length_effect(analysis, outdir)
	fig2_variance_decomposition(analysis, outdir)
	fig3_positional_preferences(analysis, outdir)
	fig4_gradcam_comparison(outdir)

	print("All figures generated.")


if __name__ == "__main__":
	main()
