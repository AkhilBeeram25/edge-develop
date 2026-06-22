"""Open-vocabulary prototype registry for novel category scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "PrototypeRegistry requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc


@dataclass(frozen=True)
class PrototypeMatch:
    names: Sequence[str]
    logits: Tensor
    probabilities: Tensor


class PrototypeRegistry(nn.Module):
    """Runtime registry for text-derived or exemplar-derived class prototypes."""

    def __init__(self, embedding_dim: int = 256, logit_scale: float = 14.2857) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.names: list[str] = []
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))
        self.register_buffer("prototypes", torch.empty(0, embedding_dim), persistent=False)

    @torch.no_grad()
    def set_prototypes(self, names: Iterable[str], prototypes: Tensor) -> None:
        names = list(names)
        if prototypes.ndim != 2 or prototypes.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected prototypes with shape [N, {self.embedding_dim}], "
                f"got {tuple(prototypes.shape)}."
            )
        if len(names) != prototypes.shape[0]:
            raise ValueError("Prototype name count must match prototype rows.")
        self.names = names
        self.prototypes = F.normalize(prototypes.detach(), dim=1)

    def forward(self, region_embeddings: Tensor) -> PrototypeMatch:
        """Score region embeddings against registered prototypes.

        Shape convention:
            region_embeddings: [..., embedding_dim]
            logits/probabilities: [..., num_registered_classes]
        """

        if self.prototypes.numel() == 0:
            empty = region_embeddings.new_empty(*region_embeddings.shape[:-1], 0)
            return PrototypeMatch(names=tuple(), logits=empty, probabilities=empty)
        embeddings = F.normalize(region_embeddings, dim=-1)
        scale = self.logit_scale.clamp(1.0, 100.0)
        logits = scale * embeddings @ self.prototypes.t()
        return PrototypeMatch(
            names=tuple(self.names),
            logits=logits,
            probabilities=logits.sigmoid(),
        )

