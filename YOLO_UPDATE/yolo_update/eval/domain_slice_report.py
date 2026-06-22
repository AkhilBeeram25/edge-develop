"""Aggregate metrics by operational domain slices."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping


@dataclass(frozen=True)
class DomainSlice:
    name: str
    fields: tuple[str, ...]

    def key_for(self, metadata: Mapping[str, object]) -> tuple[Hashable, ...]:
        return tuple(metadata.get(field, "unknown") for field in self.fields)


def group_by_domain_slice(
    records: Iterable[Mapping[str, object]],
    domain_slice: DomainSlice,
) -> dict[tuple[Hashable, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[Hashable, ...], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[domain_slice.key_for(record)].append(record)
    return dict(grouped)


def summarize_scalar_by_slice(
    records: Iterable[Mapping[str, object]],
    domain_slice: DomainSlice,
    metric_name: str,
) -> dict[tuple[Hashable, ...], dict[str, float]]:
    grouped = group_by_domain_slice(records, domain_slice)
    summary: dict[tuple[Hashable, ...], dict[str, float]] = {}
    for key, items in grouped.items():
        values = [float(item[metric_name]) for item in items if metric_name in item]
        if not values:
            summary[key] = {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        summary[key] = {
            "count": float(len(values)),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary

