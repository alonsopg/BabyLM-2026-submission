from __future__ import annotations

import hashlib
from dataclasses import dataclass


def stable_hash(key: str, seed: int, width: int) -> int:
    raw = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "little") % width


@dataclass
class CountMinSketch:
    width: int
    depth: int
    seed: int = 0
    conservative_update: bool = True

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        self.table = [[0.0 for _ in range(int(self.width))] for _ in range(int(self.depth))]
        self.total_updates = 0

    def _indices(self, key: str) -> list[tuple[int, int]]:
        return [(row, stable_hash(str(key), self.seed + row, self.width)) for row in range(self.depth)]

    def update(self, key: str, value: float = 1.0):
        if value <= 0:
            return
        key = str(key)
        indices = self._indices(key)
        if self.conservative_update:
            target = min(self.table[row][col] for row, col in indices) + float(value)
            for row, col in indices:
                if self.table[row][col] < target:
                    self.table[row][col] = target
        else:
            for row, col in indices:
                self.table[row][col] += float(value)
        self.total_updates += 1

    def estimate(self, key: str) -> float:
        return float(min(self.table[row][col] for row, col in self._indices(str(key))))

    def batch_estimate(self, keys: list[str]) -> list[float]:
        return [self.estimate(key) for key in keys]

    def reset(self):
        for row in range(self.depth):
            for col in range(self.width):
                self.table[row][col] = 0.0
        self.total_updates = 0

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "depth": self.depth,
            "seed": self.seed,
            "conservative_update": self.conservative_update,
            "total_updates": self.total_updates,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CountMinSketch":
        sketch = cls(
            width=int(data["width"]),
            depth=int(data["depth"]),
            seed=int(data.get("seed", 0)),
            conservative_update=bool(data.get("conservative_update", True)),
        )
        sketch.table = [[float(value) for value in row] for row in data["table"]]
        sketch.total_updates = int(data.get("total_updates", 0))
        return sketch
