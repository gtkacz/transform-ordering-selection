from collections.abc import Callable
from typing import TypedDict

from torch import Tensor
from torch.utils.data import Dataset

type LossFunction = Callable[[Tensor, Tensor], Tensor]
type TrainingDataset = Dataset[Tensor]
type ValidationDataset = Dataset[Tensor]
type TestingDataset = Dataset[Tensor]


class ConfusionMatrix(TypedDict):
	"""Binary classification confusion matrix with TP/TN/FP/FN counts."""

	TP: int
	TN: int
	FP: int
	FN: int
