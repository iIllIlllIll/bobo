from __future__ import annotations

from functools import lru_cache
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Dict, Type

from backtest.strategy import Strategy


def _strategies_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "strategies"


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _load_module(path: Path):
    module_name = f"strategy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def discover_strategies() -> Dict[str, Type[Strategy]]:
    strategies: Dict[str, Type[Strategy]] = {}
    directory = _strategies_dir()
    if not directory.exists():
        return strategies

    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = _load_module(path)
        if module is None:
            continue
        module_name = getattr(module, "STRATEGY_NAME", None)
        for obj in module.__dict__.values():
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, Strategy) or obj is Strategy:
                continue
            name = getattr(obj, "NAME", None) or module_name or obj.__name__
            normalized = _normalize_name(str(name))
            if normalized in strategies:
                continue
            strategies[normalized] = obj
    return strategies


def list_strategy_names() -> list[str]:
    return sorted(discover_strategies().keys())


def get_strategy_class(name: str) -> Type[Strategy]:
    strategies = discover_strategies()
    normalized = _normalize_name(name)
    if normalized not in strategies:
        available = ", ".join(list_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return strategies[normalized]
