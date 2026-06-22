"""Multi-task loss balancing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Multi-task loss balancing requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc


@dataclass(frozen=True)
class FixedLossWeights:
    detection: float = 1.0
    segmentation: float = 0.75
    image_classification: float = 0.25
    embedding: float = 0.20
    distillation: float = 0.30

    def combine(self, losses: Mapping[str, Tensor]) -> Tensor:
        total = None
        for name, weight in self.__dict__.items():
            if name not in losses:
                continue
            value = weight * losses[name]
            total = value if total is None else total + value
        if total is None:
            raise ValueError("No matching loss names were provided.")
        return total


class UncertaintyWeightedLoss(nn.Module):
    """Learnable homoscedastic uncertainty weighting with optional caps."""

    def __init__(self, task_names: tuple[str, ...], max_log_variance: float = 3.0) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("At least one task name is required.")
        self.task_names = task_names
        self.max_log_variance = max_log_variance
        self.log_variance = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in task_names}
        )

    def forward(self, losses: Mapping[str, Tensor]) -> Tensor:
        total = None
        for name in self.task_names:
            if name not in losses:
                continue
            log_var = self.log_variance[name].clamp(
                min=-self.max_log_variance,
                max=self.max_log_variance,
            )
            weighted = torch.exp(-log_var) * losses[name] + log_var
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No matching loss names were provided.")
        return total

