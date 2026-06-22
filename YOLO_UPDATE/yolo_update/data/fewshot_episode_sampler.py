"""Few-shot episode sampler for prototype-based training."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Generic, Mapping, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class FewShotEpisode(Generic[T]):
    class_name: str
    support: tuple[T, ...]
    query: tuple[T, ...]


class FewShotEpisodeSampler(Generic[T]):
    """Sample class-balanced support/query episodes from indexed examples."""

    def __init__(
        self,
        examples_by_class: Mapping[str, Sequence[T]],
        shots: int = 5,
        queries: int = 5,
        seed: int | None = None,
    ) -> None:
        if shots <= 0 or queries <= 0:
            raise ValueError("shots and queries must be positive.")
        self.examples_by_class = {name: tuple(items) for name, items in examples_by_class.items()}
        self.shots = shots
        self.queries = queries
        self.rng = random.Random(seed)
        self.valid_classes = [
            name for name, items in self.examples_by_class.items() if len(items) >= shots + queries
        ]
        if not self.valid_classes:
            raise ValueError("No class has enough examples for the requested episode shape.")

    def sample(self) -> FewShotEpisode[T]:
        class_name = self.rng.choice(self.valid_classes)
        examples = list(self.examples_by_class[class_name])
        self.rng.shuffle(examples)
        support = tuple(examples[: self.shots])
        query = tuple(examples[self.shots : self.shots + self.queries])
        return FewShotEpisode(class_name=class_name, support=support, query=query)

    def sample_many(self, count: int) -> list[FewShotEpisode[T]]:
        if count < 0:
            raise ValueError("count must be non-negative.")
        return [self.sample() for _ in range(count)]

