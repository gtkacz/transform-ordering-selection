from collections.abc import Sequence

import kornia
import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn


def apply_gpu_transforms(batch: Tensor, transforms: Sequence[nn.Module]) -> Tensor:
	"""Apply a sequence of GPU-side transforms to a batch already on device.

	Args:
		batch: Image batch of shape (B, C, H, W) already on the target device.
		transforms: Sequence of nn.Module transforms to apply in order.

	Returns:
		Transformed batch tensor.
	"""
	for t in transforms:
		batch = t(batch)
	return batch


class NormalizeTransform(nn.Module):
	"""Normalize a tensor image with mean and standard deviation (GPU-compatible via nn.Module)."""

	def __init__(
		self,
		mean: list[float] | None = None,
		std: list[float] | None = None,
	) -> None:
		"""Args:
		mean: Per-channel means; defaults to [0.5].
		std: Per-channel standard deviations; defaults to [0.5].
		"""
		super().__init__()
		mean = mean or [0.5]
		std = std or [0.5]
		self.register_buffer("mean", torch.tensor(mean))
		self.register_buffer("std", torch.tensor(std))

	def forward(self, tensor: Tensor) -> Tensor:
		"""Args:
		tensor: Image tensor of shape (C, H, W) or (B, C, H, W).

		Returns:
			Normalized tensor.
		"""
		return TF.normalize(tensor, self.mean.tolist(), self.std.tolist())


class DenoiseTransform(nn.Module):
	"""
	Apply denoising to the input image using bilateral filtering (GPU-accelerated via kornia).
	"""

	def __init__(self, h=10, template_window_size=7, search_window_size=21):
		super().__init__()
		# Map OpenCV NLM params to bilateral blur params:
		# template_window_size -> kernel_size (must be odd)
		kernel_size = template_window_size if template_window_size % 2 == 1 else template_window_size + 1
		self.kernel_size = (kernel_size, kernel_size)
		# h -> sigma_color (filter strength)
		self.sigma_color = float(h)
		# search_window_size -> sigma_space
		self.sigma_space = (float(search_window_size), float(search_window_size))

	def forward(self, img):
		if not isinstance(img, torch.Tensor):
			img = TF.to_tensor(img)

		# bilateral_blur expects (B, C, H, W)
		needs_batch = img.dim() == 3
		if needs_batch:
			img = img.unsqueeze(0)

		denoised = kornia.filters.bilateral_blur(img, self.kernel_size, self.sigma_color, self.sigma_space)

		if needs_batch:
			denoised = denoised.squeeze(0)

		return denoised


class ColorSpaceTransform(nn.Module):
	"""
	Change the color space of the input image (GPU-accelerated via kornia).
	Supported color spaces: 'RGB', 'BGR', 'HSV', 'LAB', 'YUV'
	"""

	_CONVERSIONS = {
		("RGB", "HSV"): kornia.color.rgb_to_hsv,
		("RGB", "LAB"): lambda x: kornia.color.rgb_to_lab(x),
		("HSV", "RGB"): kornia.color.hsv_to_rgb,
		("LAB", "RGB"): lambda x: kornia.color.lab_to_rgb(x),
		("RGB", "YUV"): kornia.color.rgb_to_yuv,
		("YUV", "RGB"): kornia.color.yuv_to_rgb,
	}

	def __init__(self, source_space="RGB", target_space="HSV"):
		super().__init__()
		self.source_space = source_space
		self.target_space = target_space

		if source_space == "RGB" and target_space == "BGR":
			self._convert = lambda x: x.flip(-3)
		elif source_space == "BGR" and target_space == "RGB":
			self._convert = lambda x: x.flip(-3)
		else:
			key = (source_space, target_space)
			if key not in self._CONVERSIONS:
				raise ValueError(f"Unsupported color space conversion: {source_space} to {target_space}")
			self._convert = self._CONVERSIONS[key]

	def forward(self, img):
		if not isinstance(img, torch.Tensor):
			img = TF.to_tensor(img)

		needs_batch = img.dim() == 3
		if needs_batch:
			img = img.unsqueeze(0)

		# Clamp to [0, 1] — kornia color conversions expect this range;
		# upstream transforms (e.g. normalization) may shift values outside it.
		img = img.clamp(0.0, 1.0)

		converted = self._convert(img)

		if needs_batch:
			converted = converted.squeeze(0)

		return converted


class EqualizationTransform(nn.Module):
	"""
	Apply histogram equalization to the input image (GPU-accelerated via kornia).
	Equalizes only the Y (luminance) channel in YUV space, preserving color.
	"""

	def forward(self, img):
		if not isinstance(img, torch.Tensor):
			img = TF.to_tensor(img)

		needs_batch = img.dim() == 3
		if needs_batch:
			img = img.unsqueeze(0)

		# Clamp to [0, 1] — upstream transforms (e.g. normalization) may shift
		# values outside the range that equalization and YUV conversion require.
		img = img.clamp(0.0, 1.0)

		if img.shape[-3] == 3:
			yuv = kornia.color.rgb_to_yuv(img)
			y_eq = kornia.enhance.equalize(yuv[:, 0:1, :, :])
			yuv = torch.cat([y_eq, yuv[:, 1:, :, :]], dim=1)
			equalized = kornia.color.yuv_to_rgb(yuv)
		else:
			equalized = kornia.enhance.equalize(img)

		if needs_batch:
			equalized = equalized.squeeze(0)

		return equalized
