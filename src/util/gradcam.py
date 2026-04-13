"""Grad-CAM heatmap generation for binary logit-output CNN models."""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

from .preprocessing import apply_gpu_transforms


class BinaryClassifierTarget:
	"""Target for binary logit-output models.

	pytorch-grad-cam expects a callable that extracts a scalar from the model output.
	For BinaryCNN (logit output, shape [B, 1]), returns the squeezed logit
	for the positive class, or negated logit for the negative class.
	"""

	def __init__(self, category: int = 1) -> None:
		"""Initialize the target.

		Args:
			category: 1 for positive (diseased), 0 for negative (healthy).
		"""
		self.category = category

	def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
		"""Extract the target score from model output.

		Args:
			model_output: Raw logit tensor.

		Returns:
			Scalar score for the target category.
		"""
		logits = model_output.squeeze(-1)
		if self.category == 1:
			return logits
		return -logits


def create_gradcam(model: nn.Module, target_layer_name: str = "conv4") -> GradCAM:
	"""Create a GradCAM instance targeting a specific conv block.

	Args:
		model: The neural network model.
		target_layer_name: Name of the convolutional block to target.

	Returns:
		Configured GradCAM instance.
	"""
	target_layer = getattr(model, target_layer_name)
	return GradCAM(model=model, target_layers=[target_layer])


def generate_heatmap(
	cam: GradCAM,
	input_tensor: torch.Tensor,
	target_category: int = 1,
	gpu_transforms: list[nn.Module] | None = None,
) -> np.ndarray:
	"""Generate a (H, W) GradCAM heatmap in [0, 1].

	Args:
		cam: GradCAM instance.
		input_tensor: Single image tensor of shape (C, H, W) or (1, C, H, W).
		target_category: 1 for positive (diseased), 0 for negative (healthy).
		gpu_transforms: Optional GPU-side preprocessing transforms to apply before inference.

	Returns:
		Heatmap array of shape (H, W) with values in [0, 1].
	"""
	if input_tensor.dim() == 3:
		input_tensor = input_tensor.unsqueeze(0)

	if gpu_transforms:
		input_tensor = apply_gpu_transforms(input_tensor, gpu_transforms)

	targets = [BinaryClassifierTarget(category=target_category)]
	grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
	return grayscale_cam[0]


def overlay_heatmap(heatmap: np.ndarray, original_image: np.ndarray) -> np.ndarray:
	"""Overlay a GradCAM heatmap on an RGB image.

	Args:
		heatmap: (H, W) array in [0, 1].
		original_image: (H, W, 3) RGB array in [0, 1] float32.

	Returns:
		(H, W, 3) overlaid image in [0, 1].
	"""
	rgb = original_image.astype(np.float32)
	if rgb.max() > 1.0:
		rgb /= 255.0
	return show_cam_on_image(rgb, heatmap, use_rgb=True)


def _pick_class_references(entries: list[dict], target_class: int, n: int) -> list[dict]:
	"""Pick diverse reference images for one class from collected predictions.

	Selects: highest-confidence correct, near-threshold, misclassified (or fallback).

	Args:
		entries: All entries for this class.
		target_class: The ground-truth class label (0 or 1).
		n: Maximum number of images to pick.

	Returns:
		Up to n reference image dicts.
	"""
	correct_by_conf = sorted(
		[e for e in entries if e["prediction"] == target_class],
		key=lambda x: x["confidence"] if target_class == 1 else (1 - x["confidence"]),
		reverse=True,
	)
	near_threshold = sorted(
		[e for e in entries if e["prediction"] == target_class],
		key=lambda x: abs(x["confidence"] - 0.5),
	)
	misclassified = [e for e in entries if e["prediction"] != target_class]

	picks: list[dict] = []

	if correct_by_conf:
		picks.append(correct_by_conf[0])

	if near_threshold:
		candidate = near_threshold[0]
		if not picks or candidate["image_id"] != picks[0]["image_id"]:
			picks.append(candidate)

	if misclassified:
		picks.append(misclassified[0])
	elif len(correct_by_conf) > 1:
		picks.append(correct_by_conf[-1])

	seen_ids = {p["image_id"] for p in picks}
	for entry in entries:
		if len(picks) >= n:
			break
		if entry["image_id"] not in seen_ids:
			picks.append(entry)
			seen_ids.add(entry["image_id"])

	return picks[:n]


def select_reference_images(
	predictions: list[dict],
	n_per_class: int = 3,
) -> list[dict]:
	"""Select reference images spanning the model's confidence distribution.

	For each class (healthy=0, diseased=1), selects up to n_per_class images:
	highest-confidence correct, near-threshold, and misclassified (or fallback).

	Args:
		predictions: Pre-computed per-sample predictions from
			``compute_test_metrics_with_samples``.  Each dict must contain
			keys: image, label, confidence, prediction, image_id.
		n_per_class: Number of reference images per class.

	Returns:
		List of dicts with keys: image, label, confidence, prediction, image_id.
	"""
	selected: list[dict] = []
	for target_class in [0, 1]:
		class_entries = [e for e in predictions if e["label"] == target_class]
		selected.extend(_pick_class_references(class_entries, target_class, n_per_class))

	return selected


def run_gradcam_analysis(
	model: nn.Module,
	reference_images: list[dict],
	device: torch.device,
	gpu_transforms: list[nn.Module] | None = None,
) -> list[dict]:
	"""Generate positive + negative Grad-CAM for each reference image.

	Args:
		model: Trained model.
		reference_images: Output from select_reference_images().
		device: Target device.
		gpu_transforms: Optional GPU-side preprocessing transforms.

	Returns:
		List of dicts with keys: image_id, label, confidence,
		heatmap_pos, heatmap_neg, overlay_pos, overlay_neg.
	"""
	cam = create_gradcam(model)
	results: list[dict] = []

	for ref in reference_images:
		img_tensor = ref["image"].to(device).unsqueeze(0)

		if gpu_transforms:
			img_tensor = apply_gpu_transforms(img_tensor, gpu_transforms)

		# Generate heatmaps for both categories
		heatmap_pos = generate_heatmap(cam, img_tensor, target_category=1)
		heatmap_neg = generate_heatmap(cam, img_tensor, target_category=0)

		# Create RGB numpy image for overlay
		original_np = ref["image"].permute(1, 2, 0).numpy()

		results.append({
			"image_id": ref["image_id"],
			"label": ref["label"],
			"confidence": ref["confidence"],
			"heatmap_pos": heatmap_pos,
			"heatmap_neg": heatmap_neg,
			"overlay_pos": overlay_heatmap(heatmap_pos, original_np),
			"overlay_neg": overlay_heatmap(heatmap_neg, original_np),
		})

	del cam
	return results
