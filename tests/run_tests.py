from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    failures = []
    for path in sorted(Path(__file__).resolve().parent.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_"):
                try:
                    fn()
                    print(f"PASS {path.stem}.{name}")
                except Exception as exc:
                    failures.append((path.stem, name, exc))
                    print(f"FAIL {path.stem}.{name}: {exc}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
