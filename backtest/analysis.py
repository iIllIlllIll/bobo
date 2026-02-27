from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence
import tempfile

import numpy as np
import pandas as pd

from backtest.plot import (
    create_exit_reason_holding_time_chart,
    create_exit_reason_signal_candle_chart,
)
from backtest.result import BacktestResult


@dataclass
class EntryDiffConfig:
    min_side_samples: int = 10
    top_features: int = 3
    max_groups: int = 24


_INDICATOR_TOKENS = (
    "rsi",
    "slope",
    "ema",
    "sma",
    "macd",
    "adx",
    "atr",
    "vwap",
    "stoch",
    "cci",
    "mom",
    "momentum",
    "roc",
    "boll",
    "bb",
    "keltner",
    "kama",
)


def _resolve_indicator_columns(
    data: pd.DataFrame,
    max_cols: int = 8,
    include: Optional[Sequence[str]] = None,
) -> List[str]:
    if data.empty:
        return []
    exclude = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    include_set = {str(item).lower() for item in include} if include else set()
    candidates: List[str] = []
    for col in data.columns:
        col_name = str(col)
        lower = col_name.lower()
        if lower in exclude:
            continue
        if include_set and lower in include_set:
            if pd.api.types.is_numeric_dtype(data[col]):
                candidates.append(col_name)
            continue
        if any(token in lower for token in _INDICATOR_TOKENS):
            if pd.api.types.is_numeric_dtype(data[col]):
                candidates.append(col_name)

    def _rank(col: str) -> Tuple[int, str]:
        name = col.lower()
        idx = len(_INDICATOR_TOKENS)
        for i, token in enumerate(_INDICATOR_TOKENS):
            if token in name:
                idx = i
                break
        return idx, name

    deduped = sorted(set(candidates), key=_rank)
    return deduped[: max(0, int(max_cols))]


def _atr_pct(data: pd.DataFrame, window: int = 14) -> pd.Series:
    if not {"high", "low", "close"}.issubset(data.columns):
        return pd.Series([np.nan] * len(data), index=data.index)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    denom = close.replace(0, np.nan)
    return (atr / denom) * 100.0


def _build_exit_trade_summaries(trades: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current: Dict[int, Optional[Dict[str, Any]]] = {1: None, -1: None}
    summaries: List[Dict[str, Any]] = []

    for trade in trades:
        ttype = trade.get("type")
        direction = int(trade.get("direction", 0))
        if ttype == "entry":
            current[direction] = {
                "add_count": 0,
                "entry_time": trade.get("timestamp"),
                "entry_price": trade.get("price"),
                "entry_reason": trade.get("reason") or "",
            }
        elif ttype == "add":
            state = current.get(direction)
            if state is not None:
                state["add_count"] = int(state.get("add_count", 0)) + 1
        elif ttype == "exit":
            state = current.get(direction) or {}
            summaries.append(
                {
                    "direction": direction,
                    "entry_time": trade.get("entry_time") or state.get("entry_time"),
                    "entry_price": trade.get("entry_price") or state.get("entry_price"),
                    "entry_reason": trade.get("entry_reason") or state.get("entry_reason") or "",
                    "exit_time": trade.get("timestamp"),
                    "exit_reason": trade.get("exit_reason") or "unknown",
                    "add_count": int(state.get("add_count", 0)),
                    "pnl": float(trade.get("pnl", 0.0)),
                    "return": float(trade.get("return", 0.0)),
                }
            )
            if direction in current:
                current[direction] = None

    return summaries


def build_entry_feature_frame(
    result: BacktestResult,
    data: pd.DataFrame,
    indicator_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    summaries = _build_exit_trade_summaries(result.trades)
    if not summaries:
        return pd.DataFrame()

    if "timestamp" not in data.columns:
        return pd.DataFrame()

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    index_map: Dict[pd.Timestamp, int] = {}
    for idx, ts in enumerate(timestamps):
        if pd.isna(ts):
            continue
        if ts not in index_map:
            index_map[ts] = idx

    atr_pct = _atr_pct(data)
    extra_cols = _resolve_indicator_columns(data, include=indicator_cols)

    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        entry_time = summary.get("entry_time")
        if entry_time is None:
            continue
        entry_ts = pd.to_datetime(entry_time, errors="coerce")
        if pd.isna(entry_ts):
            continue
        exit_time = summary.get("exit_time")
        exit_ts = pd.to_datetime(exit_time, errors="coerce") if exit_time is not None else pd.NaT
        idx = index_map.get(entry_ts)
        if idx is None:
            continue
        row = data.iloc[idx]
        open_price = float(row.get("open", np.nan))
        high_price = float(row.get("high", np.nan))
        low_price = float(row.get("low", np.nan))
        close_price = float(row.get("close", np.nan))
        volume = float(row.get("volume", np.nan)) if "volume" in row else np.nan
        candle_return = (close_price - open_price) / open_price * 100.0 if open_price else np.nan
        range_pct = (high_price - low_price) / open_price * 100.0 if open_price else np.nan
        body_pct = (close_price - open_price) / open_price * 100.0 if open_price else np.nan
        upper_wick = (high_price - max(open_price, close_price)) / open_price * 100.0 if open_price else np.nan
        lower_wick = (min(open_price, close_price) - low_price) / open_price * 100.0 if open_price else np.nan
        close_pos = (
            (close_price - low_price) / (high_price - low_price)
            if high_price and low_price and high_price != low_price
            else np.nan
        )
        row_payload = {
            "exit_reason": summary.get("exit_reason"),
            "add_count": int(summary.get("add_count", 0)),
            "pnl": summary.get("pnl"),
            "return": summary.get("return"),
            "win": float(summary.get("pnl", 0.0)) > 0.0,
            "direction": int(summary.get("direction", 0)),
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "entry_price": summary.get("entry_price"),
            "entry_reason": summary.get("entry_reason"),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "candle_return_pct": candle_return,
            "range_pct": range_pct,
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick,
            "lower_wick_pct": lower_wick,
            "close_pos": close_pos,
            "atr_pct": float(atr_pct.iloc[idx]) if not atr_pct.empty else np.nan,
            "hour": int(entry_ts.hour),
            "minute": int(entry_ts.minute),
            "weekday": int(entry_ts.weekday()),
        }
        if extra_cols:
            for col in extra_cols:
                if col in row_payload:
                    continue
                raw = row.get(col, np.nan)
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = np.nan
                if not np.isfinite(value):
                    value = np.nan
                row_payload[col] = value
        rows.append(row_payload)

    return pd.DataFrame(rows)


def _attach_signal_hour_stats(
    frame: pd.DataFrame, data: pd.DataFrame
) -> pd.DataFrame:
    if frame.empty or data.empty or "timestamp" not in data.columns:
        return frame
    if "entry_time" not in frame.columns:
        return frame
    price_cols = {"open", "high", "low", "close"}
    if not price_cols.issubset(data.columns):
        return frame
    data_ts = pd.to_datetime(data["timestamp"], errors="coerce")
    hourly = data.copy()
    hourly["timestamp"] = data_ts
    hourly = hourly.dropna(subset=["timestamp"]).sort_values("timestamp")
    if hourly.empty:
        return frame
    hourly = hourly.set_index("timestamp")
    hourly_ohlc = hourly[list(price_cols)].astype(float).resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    hourly_ohlc = hourly_ohlc.dropna(subset=["open", "close"])
    if hourly_ohlc.empty:
        return frame

    entries = frame.copy()
    # Use lowercase alias for compatibility with stricter pandas frequency parsing
    entries["signal_hour"] = (
        pd.to_datetime(entries["entry_time"], errors="coerce").dt.floor("h")
    )
    entries = entries.merge(
        hourly_ohlc,
        left_on="signal_hour",
        right_index=True,
        how="left",
        suffixes=("", "_signal_hour"),
    )
    open_col = entries["open_signal_hour"].astype(float)
    close_col = entries["close_signal_hour"].astype(float)
    denom = open_col.replace(0, np.nan)
    entries["signal_hour_return_pct"] = ((close_col - open_col) / denom) * 100.0
    entries["signal_hour_body_pct"] = ((close_col - open_col).abs() / denom) * 100.0
    return entries


def compute_prev_extreme_breach_stats(
    result: BacktestResult,
    data: pd.DataFrame,
    lookback: int,
    margin_pct: float,
) -> Dict[str, Any]:
    if data.empty or lookback <= 0:
        return {}
    if not {"timestamp", "low", "high"}.issubset(data.columns):
        return {}

    summaries = _build_exit_trade_summaries(result.trades)
    if not summaries:
        return {}

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    index_map: Dict[pd.Timestamp, int] = {}
    for idx, ts in enumerate(timestamps):
        if pd.isna(ts):
            continue
        if ts not in index_map:
            index_map[ts] = idx

    margin_frac = abs(float(margin_pct)) / 100.0
    totals: Dict[str, int] = {}
    breached: Dict[str, int] = {}
    evaluated = 0
    skipped = 0

    for summary in summaries:
        direction = int(summary.get("direction", 0))
        if direction not in {1, -1}:
            skipped += 1
            continue
        entry_time = summary.get("entry_time")
        if entry_time is None:
            skipped += 1
            continue
        entry_ts = pd.to_datetime(entry_time, errors="coerce")
        if pd.isna(entry_ts):
            skipped += 1
            continue
        idx = index_map.get(entry_ts)
        if idx is None or idx <= 0:
            skipped += 1
            continue
        start = max(0, idx - lookback)
        if start >= idx:
            skipped += 1
            continue
        window = data.iloc[start:idx]
        prev_low = float(window["low"].min())
        prev_high = float(window["high"].max())
        entry_row = data.iloc[idx]
        entry_low = float(entry_row.get("low", np.nan))
        entry_high = float(entry_row.get("high", np.nan))
        if not np.isfinite(prev_low) or not np.isfinite(prev_high):
            skipped += 1
            continue
        if not np.isfinite(entry_low) or not np.isfinite(entry_high):
            skipped += 1
            continue

        reason = str(summary.get("exit_reason") or "unknown")
        totals[reason] = totals.get(reason, 0) + 1
        evaluated += 1

        is_breached = False
        if direction == 1:
            threshold = prev_high * (1.0 + margin_frac)
            is_breached = entry_high >= threshold
        elif direction == -1:
            threshold = prev_low * (1.0 - margin_frac)
            is_breached = entry_low <= threshold

        if is_breached:
            breached[reason] = breached.get(reason, 0) + 1

    by_reason: Dict[str, Dict[str, float]] = {}
    for reason, total in totals.items():
        hit = breached.get(reason, 0)
        ratio = hit / total if total > 0 else 0.0
        by_reason[reason] = {"total": total, "breached": hit, "ratio": ratio}

    total_breached = sum(breached.values())
    overall_ratio = total_breached / evaluated if evaluated > 0 else 0.0
    return {
        "lookback": lookback,
        "margin_pct": margin_pct,
        "evaluated": evaluated,
        "skipped": skipped,
        "breached": total_breached,
        "ratio": overall_ratio,
        "by_reason": by_reason,
    }


def _select_feature_columns(frame: pd.DataFrame) -> List[str]:
    preferred = [
        "candle_return_pct",
        "range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "close_pos",
        "atr_pct",
        "volume",
        "hour",
        "minute",
        "weekday",
    ]
    features = [col for col in preferred if col in frame.columns]
    return [col for col in features if pd.api.types.is_numeric_dtype(frame[col])]


def _diff_for_group(
    group: pd.DataFrame,
    feature_cols: List[str],
    min_side_samples: int,
) -> Tuple[Optional[Dict[str, float]], int, int]:
    wins = group[group["win"]]
    losses = group[~group["win"]]
    win_count = len(wins)
    loss_count = len(losses)
    if win_count < min_side_samples or loss_count < min_side_samples:
        return None, win_count, loss_count
    diffs: Dict[str, float] = {}
    for feature in feature_cols:
        win_mean = wins[feature].astype(float).mean()
        loss_mean = losses[feature].astype(float).mean()
        if pd.isna(win_mean) or pd.isna(loss_mean):
            continue
        diffs[feature] = float(win_mean - loss_mean)
    return diffs, win_count, loss_count


def _stats_for_group(
    group: pd.DataFrame,
    feature_cols: List[str],
    min_side_samples: int,
) -> Tuple[Optional[Dict[str, Tuple[float, float, float]]], int, int]:
    wins = group[group["win"]]
    losses = group[~group["win"]]
    win_count = len(wins)
    loss_count = len(losses)
    if win_count < min_side_samples or loss_count < min_side_samples:
        return None, win_count, loss_count
    stats: Dict[str, Tuple[float, float, float]] = {}
    for feature in feature_cols:
        win_mean = wins[feature].astype(float).mean()
        loss_mean = losses[feature].astype(float).mean()
        if pd.isna(win_mean) or pd.isna(loss_mean):
            continue
        diff = float(win_mean - loss_mean)
        stats[feature] = (float(win_mean), float(loss_mean), diff)
    return stats, win_count, loss_count


def create_entry_diff_report(
    result: BacktestResult,
    data: pd.DataFrame,
    config: Optional[EntryDiffConfig] = None,
) -> Tuple[str, List[Path]]:
    cfg = config or EntryDiffConfig()
    frame = build_entry_feature_frame(result, data)
    if frame.empty:
        return "Entry diff: no exit trades found.", []

    frame = _attach_signal_hour_stats(frame, data)

    feature_cols = _select_feature_columns(frame)
    if not feature_cols:
        return "Entry diff: no numeric entry features available.", []

    lines: List[str] = []
    total_trades = len(frame)
    win_total = int(frame["win"].sum())
    loss_total = total_trades - win_total
    lines.append("Entry diff: entry feature means compared by exit reason and add count.")
    lines.append(f"Total exits: {total_trades} | Wins: {win_total} | Losses: {loss_total}")
    lines.append(
        f"Min samples per side: {cfg.min_side_samples} | Features: {len(feature_cols)}"
    )
    overall_stats, overall_win_count, overall_loss_count = _stats_for_group(
        frame, feature_cols, cfg.min_side_samples
    )
    if overall_stats:
        top_overall = sorted(
            overall_stats.items(), key=lambda item: abs(item[1][2]), reverse=True
        )[: max(1, int(cfg.top_features))]
        top_text = ", ".join(
            [f"{name} {vals[0]:.3f}/{vals[1]:.3f} ({vals[2]:+.3f})" for name, vals in top_overall]
        )
        lines.append(
            f"Overall top(win/loss/diff): {top_text} "
            f"| wins={overall_win_count} losses={overall_loss_count}"
        )

    group_cols = ["exit_reason", "add_count"]
    grouped = frame.groupby(group_cols, dropna=False)
    group_sizes = grouped.size().sort_values(ascending=False)

    shown = 0
    top_n = max(1, int(cfg.top_features))
    for (reason, add_count), group_size in group_sizes.items():
        if shown >= cfg.max_groups:
            break
        group = grouped.get_group((reason, add_count))
        stats, win_count, loss_count = _stats_for_group(
            group, feature_cols, cfg.min_side_samples
        )
        if stats is None:
            lines.append(
                f"- exit={reason} | add={add_count} | n={group_size} | "
                f"wins={win_count} losses={loss_count} | skipped (insufficient samples)"
            )
            shown += 1
            continue
        top = sorted(stats.items(), key=lambda item: abs(item[1][2]), reverse=True)[:top_n]
        top_text = ", ".join(
            [f"{name} {vals[0]:.3f}/{vals[1]:.3f} ({vals[2]:+.3f})" for name, vals in top]
        )
        lines.append(
            f"- exit={reason} | add={add_count} | n={group_size} | "
            f"wins={win_count} losses={loss_count} | top(win/loss/diff): {top_text}"
        )
        shown += 1

    if shown < len(group_sizes):
        lines.append(f"... ({len(group_sizes) - shown} groups not shown)")

    chart_paths: List[Path] = []
    with tempfile.NamedTemporaryFile(prefix="entry_diff_signal_hour_", suffix=".png", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)
    signal_chart = create_exit_reason_signal_candle_chart(frame, output_path)
    if signal_chart:
        chart_paths.append(signal_chart)

    with tempfile.NamedTemporaryFile(prefix="entry_diff_holding_", suffix=".png", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)
    holding_chart = create_exit_reason_holding_time_chart(frame, output_path)
    if holding_chart:
        chart_paths.append(holding_chart)

    return "\n".join(lines), chart_paths
