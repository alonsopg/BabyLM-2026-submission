from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SketchItem:
    key: str
    count: float


class SpaceSavingSketch:
    """Deterministic bounded-memory heavy-hitter sketch."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._counts: dict[str, float] = {}
        self._errors: dict[str, float] = {}
        self._order: dict[str, int] = {}
        self._clock = 0

    def update(self, key: str, weight: float = 1.0):
        if weight <= 0:
            return
        key = str(key)
        if key in self._counts:
            self._counts[key] += float(weight)
            return
        self._clock += 1
        if len(self._counts) < self.capacity:
            self._counts[key] = float(weight)
            self._errors[key] = 0.0
            self._order[key] = self._clock
            return
        victim = min(self._counts, key=lambda k: (self._counts[k], self._order[k], k))
        victim_count = self._counts.pop(victim)
        self._errors.pop(victim, None)
        self._order.pop(victim, None)
        self._counts[key] = victim_count + float(weight)
        self._errors[key] = victim_count
        self._order[key] = self._clock

    def estimate(self, key: str) -> float:
        return float(self._counts.get(str(key), 0.0))

    def contains(self, key: str) -> bool:
        return str(key) in self._counts

    def topk(self, k: int) -> list[tuple[str, float]]:
        items = sorted(self._counts.items(), key=lambda kv: (-kv[1], self._order[kv[0]], kv[0]))
        return [(key, float(count)) for key, count in items[:k]]

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "items": [{"key": k, "count": c, "error": self._errors.get(k, 0.0)} for k, c in self.topk(self.capacity)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpaceSavingSketch":
        sketch = cls(int(data["capacity"]))
        for idx, item in enumerate(data.get("items", []), start=1):
            key = str(item["key"])
            sketch._counts[key] = float(item["count"])
            sketch._errors[key] = float(item.get("error", 0.0))
            sketch._order[key] = idx
        sketch._clock = len(sketch._order)
        return sketch
