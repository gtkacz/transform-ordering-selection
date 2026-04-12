"""Evaluation metrics: test-set accuracy, confusion matrix, α/γ/wα computations."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from .types import ConfusionMatrix


def compute_test_metrics(
	model: nn.Module,
	test_loader: DataLoader,
	device: torch.device,
	use_amp: bool,
	gpu_transforms: list[nn.Module] | None = None,
) -> tuple[float, ConfusionMatrix]:
	"""Compute test accuracy and confusion matrix.

	Args:
		model: Trained neural network model (already in eval mode).
		test_loader: DataLoader for test data.
		device: Target device.
		use_amp: Whether to use automatic mixed precision.
		gpu_transforms: Optional GPU-side preprocessing transforms.

	Returns:
		Tuple of (test_accuracy, confusion_matrix).
	"""
	model.eval()
	running_corrects = 0
	total_samples = 0
	confusion_matrix: ConfusionMatrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

	with torch.no_grad():
		for images, labels in test_loader:
			images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
			labels = labels.to(device, non_blocking=True).float()

			if gpu_transforms:
				for t in gpu_transforms:
					images = t(images)

			with torch.amp.autocast("cuda", enabled=use_amp):
				outputs = model(images).squeeze()

			preds = (outputs > 0.5).float()

			confusion_matrix["TP"] += int(torch.sum((preds == 1) & (labels == 1)).item())
			confusion_matrix["TN"] += int(torch.sum((preds == 0) & (labels == 0)).item())
			confusion_matrix["FP"] += int(torch.sum((preds == 1) & (labels == 0)).item())
			confusion_matrix["FN"] += int(torch.sum((preds == 0) & (labels == 1)).item())

			running_corrects += torch.sum(preds == labels).item()
			total_samples += labels.size(0)

	accuracy = running_corrects / total_samples
	return accuracy, confusion_matrix


def compute_alpha(accuracy: float, base_accuracy: float) -> float:
	"""Compute α = accuracy - base_accuracy (absolute accuracy gain).

	Args:
		accuracy: Model accuracy with preprocessing.
		base_accuracy: Baseline model accuracy without preprocessing.

	Returns:
		The absolute accuracy gain.
	"""
	return accuracy - base_accuracy


def compute_gamma(training_time: float, base_training_time: float) -> float:
	"""Compute γ = training_time / base_training_time (time ratio).

	Args:
		training_time: Training time with preprocessing (seconds).
		base_training_time: Baseline training time without preprocessing (seconds).

	Returns:
		The training time ratio.
	"""
	return training_time / base_training_time


def compute_weighted_alpha(alpha: float, gamma: float) -> float:
	"""Compute wα = α / γ (accuracy gain per unit of training cost).

	Args:
		alpha: Absolute accuracy gain (α).
		gamma: Training time ratio (γ).

	Returns:
		The weighted alpha metric.
	"""
	return alpha / gamma
