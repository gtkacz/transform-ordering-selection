"""Configuration dataclasses and loader for the thesis pipeline."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NormalizeConfig:
	"""Parameters for NormalizeTransform."""

	mean: float
	std: float


@dataclass(frozen=True)
class DenoiseConfig:
	"""Parameters for DenoiseTransform."""

	template_window_size: int
	search_window_size: int


@dataclass(frozen=True)
class ColorSpaceConfig:
	"""Parameters for ColorSpaceTransform."""

	source_space: str
	target_space: str


@dataclass(frozen=True)
class PreprocessConfig:
	"""Full preprocessing parameter set for one confidence level."""

	normalize: NormalizeConfig
	denoise: DenoiseConfig
	colorspace: ColorSpaceConfig
	confidence_level: str


@dataclass(frozen=True)
class TrainingConfig:
	"""Training hyperparameters from parameters.toml."""

	num_epochs: int
	num_workers: int
	batch_size: int
	learning_rate: float
	shuffle: bool
	pin_memory: bool
	resize_dim: int
	precision_threshold: float


def load_configs(
	path: str | Path = "parameters.toml",
) -> tuple[TrainingConfig, PreprocessConfig, PreprocessConfig]:
	"""Load parameters.toml and return (training, base_preprocess, high_con_preprocess).

	Args:
		path: Path to the TOML configuration file.

	Returns:
		A 3-tuple of (TrainingConfig, base PreprocessConfig, high_con PreprocessConfig).
	"""
	raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

	training = TrainingConfig(
		num_epochs=raw["TRAINING"]["num_epochs"],
		num_workers=raw["TRAINING"]["num_workers"],
		batch_size=raw["TRAINING"]["batch_size"],
		learning_rate=raw["TRAINING"]["learning_rate"],
		shuffle=raw["TRAINING"]["shuffle"],
		pin_memory=raw["TRAINING"]["pin_memory"],
		resize_dim=raw["TRAINING"]["resize_dim"],
		precision_threshold=raw["TRAINING"]["precision_threshold"],
	)

	preprocess = raw["PREPROCESS"]

	base = PreprocessConfig(
		normalize=NormalizeConfig(
			mean=preprocess["normalize"]["mean"],
			std=preprocess["normalize"]["std"],
		),
		denoise=DenoiseConfig(
			template_window_size=preprocess["denoise"]["template_window_size"],
			search_window_size=preprocess["denoise"]["search_window_size"],
		),
		colorspace=ColorSpaceConfig(
			source_space=preprocess["colorspace"]["source_space"],
			target_space=preprocess["colorspace"]["target_space"],
		),
		confidence_level="base",
	)

	high_con = PreprocessConfig(
		normalize=NormalizeConfig(
			mean=preprocess["normalize"]["high_con"]["mean"],
			std=preprocess["normalize"]["high_con"]["std"],
		),
		denoise=DenoiseConfig(
			template_window_size=preprocess["denoise"]["high_con"]["template_window_size"],
			search_window_size=preprocess["denoise"]["high_con"]["search_window_size"],
		),
		colorspace=ColorSpaceConfig(
			source_space=preprocess["colorspace"]["high_con"]["source_space"],
			target_space=preprocess["colorspace"]["high_con"]["target_space"],
		),
		confidence_level="high_con",
	)

	return training, base, high_con
