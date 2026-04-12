"""Thesis pipeline: train, evaluate, and export results for all preprocessing combinations."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from util.config import load_configs
from util.export import build_results_matrix
from util.runner import run_full_experiment

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def main() -> None:
	"""Run the full thesis experiment pipeline."""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Using device: %s", device)

	training_config, base_config, high_con_config = load_configs()

	all_results = run_full_experiment(
		training_config=training_config,
		preprocess_configs=[base_config, high_con_config],
		device=device,
	)

	build_results_matrix(all_results, output_dir=OUTPUT_DIR)
	logger.info("Pipeline complete. Results written to %s/", OUTPUT_DIR)


if __name__ == "__main__":
	main()
