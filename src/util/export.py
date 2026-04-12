"""Structured export: per-combination metrics/images + master results matrix."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .metrics import compute_alpha, compute_gamma, compute_weighted_alpha
from .runner import BASELINE_KEY, CombinationResult

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


def _save_overlay_png(overlay: np.ndarray, path: Path) -> None:
	"""Save a [0, 1] float32 RGB overlay as a PNG file."""
	rgb_uint8 = (overlay * 255).clip(0, 255).astype(np.uint8)
	Image.fromarray(rgb_uint8).save(path)


def export_combination_result(
	result: CombinationResult,
	base_accuracy: float,
	base_training_time: float,
	output_dir: Path,
) -> dict:
	"""Export one (combo, confidence) pair to disk.

	Writes metrics.json and Grad-CAM overlay PNGs to
	output/{confidence}/{combo_key}/.

	Args:
		result: A single CombinationResult from the pipeline.
		base_accuracy: Baseline accuracy for α computation.
		base_training_time: Baseline training time for γ computation.
		output_dir: Root output directory.

	Returns:
		Dict entry suitable for inclusion in the master results matrix.
	"""
	alpha = compute_alpha(result.accuracy, base_accuracy)
	gamma = compute_gamma(result.training_time, base_training_time)
	weighted_alpha = compute_weighted_alpha(alpha, gamma)

	combo_dir = output_dir / result.confidence_level / result.combo_key
	combo_dir.mkdir(parents=True, exist_ok=True)

	gradcam_entries: list[dict] = []
	for gc in result.gradcam_results:
		image_id = gc["image_id"]

		pos_filename = f"gradcam_pos_{image_id}.png"
		neg_filename = f"gradcam_neg_{image_id}.png"

		_save_overlay_png(gc["overlay_pos"], combo_dir / pos_filename)
		_save_overlay_png(gc["overlay_neg"], combo_dir / neg_filename)

		gradcam_entries.append({
			"image_id": image_id,
			"label": gc["label"],
			"confidence": gc["confidence"],
			"pos_overlay": pos_filename,
			"neg_overlay": neg_filename,
		})

	metrics_entry = {
		"combo_key": result.combo_key,
		"confidence_level": result.confidence_level,
		"transforms": list(result.transforms),
		"accuracy": result.accuracy,
		"alpha": alpha,
		"gamma": gamma,
		"weighted_alpha": weighted_alpha,
		"confusion_matrix": dict(result.confusion_matrix),
		"training_time_seconds": result.training_time,
		"gradcam_images": gradcam_entries,
	}

	metrics_path = combo_dir / "metrics.json"
	metrics_path.write_text(json.dumps(metrics_entry, indent=2))

	return metrics_entry


def build_results_matrix(
	all_results: dict[str, dict[str, CombinationResult]],
	output_dir: Path,
) -> None:
	"""Export all results and write the master results_matrix.json.

	For each confidence level, the Baseline entry is used to derive
	base_accuracy and base_training_time for α/γ/wα computations.

	Args:
		all_results: Nested dict {confidence_level: {combo_key: CombinationResult}}.
		output_dir: Root output directory.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	matrix_entries: list[dict] = []

	for confidence_level, combo_results in all_results.items():
		baseline = combo_results.get(BASELINE_KEY)
		if baseline is None:
			logger.warning(
				"No baseline found for confidence level %s; skipping α/γ/wα",
				confidence_level,
			)
			base_accuracy = 0.0
			base_training_time = 1.0
		else:
			base_accuracy = baseline.accuracy
			base_training_time = baseline.training_time

		for result in combo_results.values():
			entry = export_combination_result(
				result=result,
				base_accuracy=base_accuracy,
				base_training_time=base_training_time,
				output_dir=output_dir,
			)
			matrix_entries.append(entry)

	matrix_path = output_dir / "results_matrix.json"
	matrix_path.write_text(json.dumps(matrix_entries, indent=2))

	logger.info(
		"Results matrix written: %s (%d entries)",
		matrix_path,
		len(matrix_entries),
	)
