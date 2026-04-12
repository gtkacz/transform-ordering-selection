"""Hyperparameter grid search for individual preprocessing transforms."""

from __future__ import annotations

import itertools
import json
import logging
import operator
from pathlib import Path

import numpy as np
import torch

from util.config import load_configs
from util.dataset import get_data_loaders
from util.preprocessing import ColorSpaceTransform, DenoiseTransform, NormalizeTransform
from util.runner import evaluate_model

logger = logging.getLogger(__name__)

PARAMS_DIR = Path("params")


def _search_colorspace(
	training_config: object,
	device: torch.device,
) -> list[dict]:
	"""Grid search over color space target spaces.

	Returns:
		Sorted list of result dicts (highest accuracy first).
	"""
	target_spaces = ["BGR", "HSV", "LAB", "YUV"]
	results: list[dict] = []

	for target_space in target_spaces:
		logger.info("Colorspace: target_space=%s", target_space)
		gpu_transforms = [ColorSpaceTransform(source_space="RGB", target_space=target_space)]
		train_loader, test_loader, val_loader = get_data_loaders(training_config)

		accuracy, confusion_matrix, training_time, _ = evaluate_model(
			device=device,
			train_loader=train_loader,
			test_loader=test_loader,
			validation_loader=val_loader,
			training_config=training_config,
			verbose=False,
			gpu_transforms=gpu_transforms,
		)

		results.append({
			"target_space": target_space,
			"accuracy": accuracy,
			"confusion_matrix": dict(confusion_matrix),
			"training_time": training_time,
		})

		torch.cuda.empty_cache()

	results.sort(key=operator.itemgetter("accuracy"), reverse=True)
	return results


def _search_denoise(
	training_config: object,
	device: torch.device,
) -> list[dict]:
	"""Grid search over denoise template_window_size and search_window_size.

	Returns:
		Sorted list of result dicts (highest accuracy first).
	"""
	combinations = list(itertools.product(range(5, 11), range(20, 26)))
	results: list[dict] = []

	for template_ws, search_ws in combinations:
		logger.info("Denoise: template=%d, search=%d", template_ws, search_ws)
		gpu_transforms = [DenoiseTransform(template_window_size=template_ws, search_window_size=search_ws)]
		train_loader, test_loader, val_loader = get_data_loaders(training_config)

		accuracy, confusion_matrix, training_time, _ = evaluate_model(
			device=device,
			train_loader=train_loader,
			test_loader=test_loader,
			validation_loader=val_loader,
			training_config=training_config,
			verbose=False,
			gpu_transforms=gpu_transforms,
		)

		results.append({
			"template_window_size": template_ws,
			"search_window_size": search_ws,
			"accuracy": accuracy,
			"confusion_matrix": dict(confusion_matrix),
			"training_time": training_time,
		})

		torch.cuda.empty_cache()

	results.sort(key=operator.itemgetter("accuracy"), reverse=True)
	return results


def _search_normalize(
	training_config: object,
	device: torch.device,
) -> list[dict]:
	"""Grid search over normalize mean and std.

	Returns:
		Sorted list of result dicts (highest accuracy first).
	"""
	combinations = list(itertools.product(np.arange(0.15, 1, 0.15), np.arange(0.15, 1, 0.15)))
	results: list[dict] = []

	for mean, std in combinations:
		mean_r = round(float(mean), 2)
		std_r = round(float(std), 2)
		logger.info("Normalize: mean=%.2f, std=%.2f", mean_r, std_r)
		gpu_transforms = [NormalizeTransform(mean=[mean_r], std=[std_r])]
		train_loader, test_loader, val_loader = get_data_loaders(training_config)

		accuracy, confusion_matrix, training_time, _ = evaluate_model(
			device=device,
			train_loader=train_loader,
			test_loader=test_loader,
			validation_loader=val_loader,
			training_config=training_config,
			verbose=False,
			gpu_transforms=gpu_transforms,
		)

		results.append({
			"mean": mean_r,
			"std": std_r,
			"accuracy": accuracy,
			"confusion_matrix": dict(confusion_matrix),
			"training_time": training_time,
		})

		torch.cuda.empty_cache()

	results.sort(key=operator.itemgetter("accuracy"), reverse=True)
	return results


def main() -> None:
	"""Run all hyperparameter grid searches and save results."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Using device: %s", device)

	training_config, *_ = load_configs()

	PARAMS_DIR.mkdir(exist_ok=True)

	logger.info("=== Colorspace grid search ===")
	cs_results = _search_colorspace(training_config, device)
	(PARAMS_DIR / "colorspace.json").write_text(json.dumps(cs_results, indent=2))

	logger.info("=== Denoise grid search ===")
	d_results = _search_denoise(training_config, device)
	(PARAMS_DIR / "denoise.json").write_text(json.dumps(d_results, indent=2))

	logger.info("=== Normalize grid search ===")
	n_results = _search_normalize(training_config, device)
	(PARAMS_DIR / "normalize.json").write_text(json.dumps(n_results, indent=2))

	logger.info("Grid search complete. Results written to %s/", PARAMS_DIR)


if __name__ == "__main__":
	main()
