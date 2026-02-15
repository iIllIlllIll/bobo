from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Settings:
    data: Dict[str, Any]
    backtest: Dict[str, Any]
    strategy: Dict[str, Any]
    discord: Dict[str, Any]


def load_settings(path: str | Path) -> Settings:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(
        data=raw.get("data", {}),
        backtest=raw.get("backtest", {}),
        strategy=raw.get("strategy", {}),
        discord=raw.get("discord", {}),
    )


def save_settings(path: str | Path, settings: Settings) -> None:
    path = Path(path)
    raw = {
        "data": settings.data,
        "backtest": settings.backtest,
        "strategy": settings.strategy,
        "discord": settings.discord,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
