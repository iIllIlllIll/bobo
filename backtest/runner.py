from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Callable

import pandas as pd
import yaml

from backtest.config import Settings, load_settings
from backtest.data import load_price_data
from backtest.engine import BacktestEngine
from backtest.analysis import EntryDiffConfig, create_entry_diff_report
from backtest.plot import (
    create_backtest_chart,
    create_equity_price_chart,
    create_stats_chart,
    create_trade_return_chart,
    create_trade_distribution_chart,
    create_holding_time_chart,
    create_daily_return_heatmaps,
)
from backtest.strategy_loader import get_strategy_class
from backtest.result import BacktestResult


def _merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _normalize_cheatkey_params(params: Dict[str, Any]) -> Dict[str, Any]:
    cheatkey_param_names = {
        "ema_fast",
        "ema_slow",
        "fast_window",
        "slow_window",
        "cheatkey_threshold",
        "cheatkey_threshold_use_pct",
        "cheatkey_threshold_pct",
        "cheatkey_lookback",
        "cheatkey_timefilter",
        "cheatkey_diff_lookback",
        "diff_lookback_n",
        "slope_exit_mode",
        "slope_exit",
        "slope_exit_lookback",
        "slope_exit_cooldown_bars",
        "slope_exit_pnl_threshold",
        "slope_exit_use_threshold",
        "slope_exit_ignore_threshold_after_add",
        "long_leverage",
        "short_leverage",
        "long_tp_pnl",
        "short_tp_pnl",
        "long_add_tp_pnl",
        "short_add_tp_pnl",
        "long_sl_pnl",
        "short_sl_pnl",
        "long_add_sl_pnl",
        "short_add_sl_pnl",
        "move_sl_on_profit",
        "move_sl_trigger_pnl",
        "move_sl_target_pnl",
        "move_sl_use_ratio",
        "move_sl_trigger_ratio",
        "move_sl_target_ratio",
        "move_tp_on_loss",
        "move_tp_trigger_pnl",
        "move_tp_target_pnl",
        "use_rr_tp_from_sl_pnl",
        "rr_tp_from_sl_multiplier",
        "long_partial_exit_pnl",
        "short_partial_exit_pnl",
        "partial_exit_mode",
        "long_add_buy_pnl",
        "short_add_buy_pnl",
        "add_buy_candle_condition",
        "use_add_buy",
        "add_buy_max_times",
        "add_buy_use_rr_ratio",
        "add_buy_rr_ratio",
        "use_second_buy",
        "divide_value",
        "total_parts",
        "exit_mode",
        "exit_full_condition",
        "first_opposite_exit",
        "first_opposite_exit_pnl",
        "exitmode_timefilter",
        "use_timeout_cross",
        "timeout_cross_bars",
        "timeout_sell_pnl",
        "use_prev_extreme_stop",
        "prev_extreme_lookback",
        "prev_extreme_buffer_pct",
        "prev_extreme_max_loss_pct",
        "use_prev_extreme_rr",
        "prev_extreme_rr_lookback",
        "prev_extreme_rr_buffer_pct",
        "prev_extreme_rr_multiplier",
        "prev_extreme_rr_max_loss_pct",
        "use_prev_extreme_rr_tp_cap",
        "prev_extreme_rr_tp_cap_lookback",
        "prev_extreme_rr_tp_cap_buffer_pct",
        "macro_ema_filter",
        "macro_ema_window",
        "macro_ema_period",
        "macro_ema_align_filter",
        "macro_ema_align_windows",
        "ema_dir_filter",
        "ema_dir_window",
        "ema_dir_lookback",
        "use_4h_candle_condition",
        "use_first_add_macro_ema",
        "first_add_candle_condition_enabled",
        "use_override_atr",
        "atr_override_period",
        "atr_pct_threshold",
        "unit_override",
        "unit",
        "unit_leverage",
        "unit_tp",
        "unit_add_tp",
        "unit_sl",
        "unit_add_sl",
        "unit_partial_exit",
        "unit_add_buy",
        "unit_timeout_sell",
        "unit_slope_exit_pnl",
        "unit_first_opposite_exit",
        "bb_width_window",
        "bb_width_std",
        "bb_width_threshold",
    }

    normalized = dict(params)
    for key, value in params.items():
        lower = key.lower()
        if lower in cheatkey_param_names and lower not in normalized:
            normalized[lower] = value
    if "diff_lookback_n" in normalized and "cheatkey_diff_lookback" not in normalized:
        normalized["cheatkey_diff_lookback"] = normalized["diff_lookback_n"]
    if "cheatkey_diff_lookback" not in normalized:
        normalized["cheatkey_diff_lookback"] = 1
    return _apply_unit_params(normalized)


def _apply_unit_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if "unit" not in params:
        return params

    normalized = dict(params)
    try:
        unit = float(normalized["unit"])
    except (TypeError, ValueError):
        return normalized

    def _set_long_short(name: str, value: Any) -> None:
        if name in normalized or value is None:
            return
        normalized[name] = value

    def _apply_to_long_short(mult_key: str, target_long: str, target_short: str, sign: float) -> None:
        if mult_key not in normalized:
            return
        mult = normalized.get(mult_key)
        if mult is None:
            return
        if isinstance(mult, list):
            values = [sign * unit * float(item) for item in mult]
            normalized[target_long] = list(values)
            normalized[target_short] = list(values)
            return
        try:
            multiplier = float(mult)
        except (TypeError, ValueError):
            return
        normalized[target_long] = sign * unit * multiplier
        normalized[target_short] = sign * unit * multiplier

    def _apply_list(mult_key: str, target_long: str, target_short: str, sign: float) -> None:
        if mult_key not in normalized:
            return
        mult = normalized.get(mult_key)
        if mult is None:
            return
        if isinstance(mult, list):
            values = [sign * unit * float(item) for item in mult]
            normalized[target_long] = list(values)
            normalized[target_short] = list(values)
            return
        try:
            multiplier = float(mult)
        except (TypeError, ValueError):
            return
        existing = normalized.get(target_long) or normalized.get(target_short)
        length = 3
        if isinstance(existing, list) and existing:
            length = len(existing)
        values = [sign * unit * multiplier for _ in range(length)]
        normalized[target_long] = list(values)
        normalized[target_short] = list(values)

    if "unit_leverage" in normalized:
        try:
            multiplier = float(normalized["unit_leverage"])
        except (TypeError, ValueError):
            multiplier = None
        if multiplier is not None:
            normalized["long_leverage"] = unit * multiplier
            normalized["short_leverage"] = unit * multiplier
    else:
        _set_long_short("long_leverage", unit)
        _set_long_short("short_leverage", unit)

    _apply_to_long_short("unit_tp", "long_tp_pnl", "short_tp_pnl", 1.0)
    _apply_list("unit_add_tp", "long_add_tp_pnl", "short_add_tp_pnl", 1.0)
    _apply_to_long_short("unit_sl", "long_sl_pnl", "short_sl_pnl", -1.0)
    _apply_list("unit_add_sl", "long_add_sl_pnl", "short_add_sl_pnl", -1.0)
    _apply_to_long_short("unit_partial_exit", "long_partial_exit_pnl", "short_partial_exit_pnl", 1.0)
    _apply_to_long_short("unit_add_buy", "long_add_buy_pnl", "short_add_buy_pnl", -1.0)

    if "unit_timeout_sell" in normalized:
        try:
            normalized["timeout_sell_pnl"] = unit * float(normalized["unit_timeout_sell"])
        except (TypeError, ValueError):
            pass
    if "unit_slope_exit_pnl" in normalized:
        try:
            normalized["slope_exit_pnl_threshold"] = unit * float(
                normalized["unit_slope_exit_pnl"]
            )
        except (TypeError, ValueError):
            pass
    if "unit_first_opposite_exit" in normalized:
        try:
            normalized["first_opposite_exit_pnl"] = unit * float(
                normalized["unit_first_opposite_exit"]
            )
        except (TypeError, ValueError):
            pass

    for key in (
        "unit",
        "unit_leverage",
        "unit_tp",
        "unit_add_tp",
        "unit_sl",
        "unit_add_sl",
        "unit_partial_exit",
        "unit_add_buy",
        "unit_timeout_sell",
        "unit_slope_exit_pnl",
        "unit_first_opposite_exit",
    ):
        normalized.pop(key, None)

    return normalized


def _attach_chart_indicators(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    if data.empty or "close" not in data.columns:
        return data

    def _ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False, min_periods=window).mean()

    fast = params.get("ema_fast") or params.get("fast_window")
    slow = params.get("ema_slow") or params.get("slow_window")
    macro = params.get("macro_ema_period") or params.get("macro_ema_window")

    df = data.copy()
    for name, window in (("ema_fast", fast), ("ema_slow", slow), ("macro_ema", macro)):
        if window is None:
            continue
        try:
            win = int(window)
        except (TypeError, ValueError):
            continue
        if win <= 0 or name in df.columns:
            continue
        df[name] = _ema(df["close"].astype(float), win)
    return df


def _parse_data_filename(filename: str) -> Optional[tuple[str, str, str, str]]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    interval = parts[-3]
    start = parts[-2]
    end = parts[-1]
    if not (start.isdigit() and end.isdigit() and len(start) == 8 and len(end) == 8):
        return None
    symbol = "_".join(parts[:-3])
    if not symbol:
        return None
    return symbol, interval, start, end


def _summarize_data_files(data_folder: str | Path, file_pattern: str) -> tuple[str, str, str]:
    folder = Path(data_folder)
    files = sorted(folder.glob(file_pattern))
    parsed = []
    for fp in files:
        info = _parse_data_filename(fp.name)
        if info:
            parsed.append(info)
    if not parsed:
        return "Unknown", "Unknown", "Unknown"

    symbols = {item[0] for item in parsed}
    intervals = {item[1] for item in parsed}
    starts = [item[2] for item in parsed]
    ends = [item[3] for item in parsed]

    symbol = symbols.pop() if len(symbols) == 1 else "mixed"
    interval = intervals.pop() if len(intervals) == 1 else "mixed"
    period = f"{min(starts)} -> {max(ends)}"
    return symbol, interval, period


def _write_trade_log(result: BacktestResult) -> Optional[Path]:
    trades = result.trades
    with tempfile.NamedTemporaryFile(prefix="backtest_trades_", suffix=".csv", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)
    if not trades:
        output_path.write_text("no trades\n", encoding="utf-8")
        return output_path
    fieldnames: List[str] = sorted({key for trade in trades for key in trade.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)
    return output_path


def _write_settings_snapshot(
    data_cfg: Dict[str, Any],
    backtest_cfg: Dict[str, Any],
    strat_name: str,
    strat_params: Dict[str, Any],
) -> Path:
    snapshot = {
        "data": data_cfg,
        "backtest": backtest_cfg,
        "strategy": {"name": strat_name, "params": strat_params},
    }
    with tempfile.NamedTemporaryFile(
        prefix="backtest_settings_", suffix=".yaml", delete=False
    ) as tmp_file:
        output_path = Path(tmp_file.name)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)
    return output_path


def create_settings_snapshot(
    settings: Settings,
    data_overrides: Optional[Dict[str, Any]] = None,
    backtest_overrides: Optional[Dict[str, Any]] = None,
    strategy_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    data_cfg = _merge_config(settings.data, data_overrides)
    strat_cfg = _merge_config(settings.strategy, strategy_overrides)
    strat_name = strat_cfg.get("name", "sma_cross")
    strat_params = dict(strat_cfg.get("params", {}))
    if strategy_overrides and "params" in strategy_overrides:
        strat_params.update(strategy_overrides.get("params", {}))
    if str(strat_name).strip().lower().replace(" ", "_").replace("-", "_") == "cheatkey":
        strat_params = _normalize_cheatkey_params(strat_params)
    bt_cfg = _merge_config(settings.backtest, backtest_overrides)
    return _write_settings_snapshot(
        data_cfg=data_cfg,
        backtest_cfg=bt_cfg,
        strat_name=strat_name,
        strat_params=strat_params,
    )


def _run_backtest(
    settings: Settings,
    data_overrides: Optional[Dict[str, Any]] = None,
    backtest_overrides: Optional[Dict[str, Any]] = None,
    strategy_overrides: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, float, List[Path], BacktestResult, pd.DataFrame]:
    data_cfg = _merge_config(settings.data, data_overrides)
    data_folder = data_cfg.get("folder", "price data")
    data_pattern = data_cfg.get("file_pattern", "*.csv")
    data_symbol, data_interval, data_period = _summarize_data_files(
        data_folder,
        data_pattern,
    )
    data = load_price_data(
        folder=data_folder,
        file_pattern=data_pattern,
        timestamp_column=data_cfg.get("timestamp_column", "timestamp"),
        timezone=data_cfg.get("timezone", "UTC"),
        datetime_format=data_cfg.get("datetime_format"),
    )

    strat_cfg = _merge_config(settings.strategy, strategy_overrides)
    strat_name = strat_cfg.get("name", "sma_cross")
    strat_params = dict(strat_cfg.get("params", {}))
    if strategy_overrides and "params" in strategy_overrides:
        strat_params.update(strategy_overrides.get("params", {}))
    if str(strat_name).strip().lower().replace(" ", "_").replace("-", "_") == "cheatkey":
        strat_params = _normalize_cheatkey_params(strat_params)
    strategy_cls = get_strategy_class(strat_name)
    data = _attach_chart_indicators(data, strat_params)
    strategy = strategy_cls(**strat_params)

    bt_cfg = _merge_config(settings.backtest, backtest_overrides)
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_balance=bt_cfg.get("initial_balance", 10000.0),
        fee_rate=bt_cfg.get("fee_rate", 0.0004),
        leverage=bt_cfg.get("leverage", 1.0),
        position_size=bt_cfg.get("position_size", 1.0),
        max_position_fraction_per_side=bt_cfg.get("max_position_fraction_per_side"),
        risk_per_trade=bt_cfg.get("risk_per_trade"),
        slippage=bt_cfg.get("slippage", 0.0),
    )

    result = engine.run(progress_cb=progress_cb)
    stats = result.stats()
    total_return = stats["total_return"] * 100.0
    max_drawdown = stats["max_drawdown"] * 100.0
    win_rate = stats["win_rate"] * 100.0
    profit_factor = stats["profit_factor"]

    if not result.equity_curve.empty:
        start_ts = result.equity_curve["timestamp"].iloc[0]
        end_ts = result.equity_curve["timestamp"].iloc[-1]
        period = f"{start_ts} -> {end_ts}"
    else:
        period = "N/A"

    profit_factor_text = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    avg_return_pct = stats["avg_return"] * 100.0
    avg_min_return_pct = stats["avg_min_return"] * 100.0
    avg_max_return_pct = stats["avg_max_return"] * 100.0
    long_ratio_pct = stats["long_ratio"] * 100.0
    short_ratio_pct = stats["short_ratio"] * 100.0
    long_avg_return_pct = stats["long_avg_return"] * 100.0
    short_avg_return_pct = stats["short_avg_return"] * 100.0
    max_win_streak = stats.get("max_win_streak", 0)
    max_loss_streak = stats.get("max_loss_streak", 0)
    def _format_reason_block(label: str, counts: Dict[str, int], total: int) -> str:
        if not counts or total <= 0:
            return ""
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        reason_parts = [
            f"{reason}={count} ({(count / total) * 100:.1f}%)" for reason, count in ordered
        ]
        return f"{label} {total} | " + ", ".join(reason_parts) + "\n"

    exit_reason_summary = ""
    exit_reason_counts_full = stats.get("exit_reason_counts_full", {})
    exit_reason_counts_partial = stats.get("exit_reason_counts_partial", {})
    exit_reason_counts_all = stats.get("exit_reason_counts", {})
    total_full = int(stats.get("exit_reason_total_full", 0))
    total_partial = int(stats.get("exit_reason_total_partial", 0))
    total_all = total_full + total_partial

    if isinstance(exit_reason_counts_all, dict) and exit_reason_counts_all:
        exit_reason_summary += _format_reason_block("🧾 Exit reasons (all):", exit_reason_counts_all, total_all)
    if isinstance(exit_reason_counts_full, dict) and exit_reason_counts_full:
        exit_reason_summary += _format_reason_block("📤 Full exits:", exit_reason_counts_full, total_full)
    if isinstance(exit_reason_counts_partial, dict) and exit_reason_counts_partial:
        exit_reason_summary += _format_reason_block(
            "📉 Partial exits:", exit_reason_counts_partial, total_partial
        )

    def _format_add_buy_distribution(distribution: Dict[int, int]) -> str:
        if not distribution:
            return ""
        ordered = sorted(distribution.items(), key=lambda item: item[0])
        parts = [f"{count}x={total}" for count, total in ordered]
        return "📌 Add count distribution: " + ", ".join(parts) + "\n"

    def _format_add_buy_win_rates(win_rates: Dict[int, Dict[str, float]]) -> str:
        if not win_rates:
            return ""
        ordered = sorted(win_rates.items(), key=lambda item: item[0])
        parts = []
        for count, bucket in ordered:
            total = int(bucket.get("total", 0))
            win_rate = float(bucket.get("win_rate", 0.0)) * 100.0
            parts.append(f"{count}x={win_rate:.1f}% (n={total})")
        return "🏁 Win rate by add count: " + ", ".join(parts) + "\n"

    def _format_add_buy_exit_reasons(reasons: Dict[int, Dict[str, int]]) -> str:
        if not reasons:
            return ""
        ordered = sorted(reasons.items(), key=lambda item: item[0])
        blocks = []
        for count, reason_counts in ordered:
            ordered_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
            top = ordered_reasons[:3]
            parts = [f"{reason}={cnt}" for reason, cnt in top]
            blocks.append(f"{count}x " + ", ".join(parts))
        return "🧾 Exit reasons by add count: " + " | ".join(blocks) + "\n"

    def _format_add_buy_debug(debug: Dict[str, Any]) -> str:
        if not debug:
            return ""
        attempts = int(debug.get("attempts", 0))
        triggered = int(debug.get("triggered", 0))
        executed = int(debug.get("executed", 0))
        blocked = debug.get("blocked", {}) if isinstance(debug.get("blocked"), dict) else {}
        blocked_total = sum(int(v) for v in blocked.values()) if blocked else 0
        attempts_by_dir = (
            debug.get("attempts_by_direction", {})
            if isinstance(debug.get("attempts_by_direction"), dict)
            else {}
        )
        triggered_by_dir = (
            debug.get("triggered_by_direction", {})
            if isinstance(debug.get("triggered_by_direction"), dict)
            else {}
        )
        executed_by_dir = (
            debug.get("executed_by_direction", {})
            if isinstance(debug.get("executed_by_direction"), dict)
            else {}
        )
        blocked_by_dir = (
            debug.get("blocked_by_direction", {})
            if isinstance(debug.get("blocked_by_direction"), dict)
            else {}
        )
        summary = (
            "🧪 Add-buy debug: "
            f"attempts={attempts} | triggered={triggered} | executed={executed} | "
            f"blocked={blocked_total}\n"
        )
        if attempts_by_dir or triggered_by_dir or executed_by_dir:
            long_attempts = int(attempts_by_dir.get(1, 0))
            short_attempts = int(attempts_by_dir.get(-1, 0))
            long_triggered = int(triggered_by_dir.get(1, 0))
            short_triggered = int(triggered_by_dir.get(-1, 0))
            long_executed = int(executed_by_dir.get(1, 0))
            short_executed = int(executed_by_dir.get(-1, 0))
            summary += (
                "🧭 By direction: "
                f"L att/tri/exec={long_attempts}/{long_triggered}/{long_executed} | "
                f"S att/tri/exec={short_attempts}/{short_triggered}/{short_executed}\n"
            )
        if blocked:
            ordered = sorted(blocked.items(), key=lambda item: (-int(item[1]), item[0]))[:6]
            parts = [
                f"{reason}={count} ({(int(count) / blocked_total) * 100:.1f}%)"
                for reason, count in ordered
                if blocked_total > 0
            ]
            if parts:
                summary += "🚫 Blocked reasons: " + ", ".join(parts) + "\n"
        if blocked_by_dir:
            for direction, reason_counts in ((1, blocked_by_dir.get(1)), (-1, blocked_by_dir.get(-1))):
                if not isinstance(reason_counts, dict) or not reason_counts:
                    continue
                total_dir = sum(int(v) for v in reason_counts.values())
                if total_dir <= 0:
                    continue
                ordered_dir = sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:4]
                parts = [
                    f"{reason}={count} ({(int(count) / total_dir) * 100:.1f}%)"
                    for reason, count in ordered_dir
                ]
                side = "Long" if direction == 1 else "Short"
                summary += f"🚫 Blocked ({side}): " + ", ".join(parts) + "\n"
        return summary

    add_buy_summary = ""
    add_buy_trade_count = int(stats.get("add_buy_trade_count", 0))
    add_buy_trade_ratio = float(stats.get("add_buy_trade_ratio", 0.0)) * 100.0
    add_buy_avg_all = float(stats.get("add_buy_avg_times_all", 0.0))
    add_buy_avg_with = float(stats.get("add_buy_avg_times_with_add", 0.0))
    add_buy_max_times = int(stats.get("add_buy_max_times", 0))
    add_buy_distribution = stats.get("add_buy_count_distribution", {})
    add_buy_win_rates = stats.get("add_buy_win_rate_by_times", {})
    add_buy_exit_reasons = stats.get("add_buy_exit_reason_counts_by_times", {})
    add_buy_debug = stats.get("add_buy_debug", {})

    add_buy_debug_ready = (
        isinstance(add_buy_debug, dict)
        and (
            int(add_buy_debug.get("attempts", 0)) > 0
            or int(add_buy_debug.get("triggered", 0)) > 0
            or int(add_buy_debug.get("executed", 0)) > 0
            or bool(add_buy_debug.get("blocked"))
        )
    )

    if stats["total_trades"] > 0:
        add_buy_summary += (
            "➕ Add-buys: "
            f"{add_buy_trade_count}/{stats['total_trades']} ({add_buy_trade_ratio:.1f}%) | "
            f"avg adds(all/added): {add_buy_avg_all:.2f}/{add_buy_avg_with:.2f} | "
            f"max adds: {add_buy_max_times}\n"
        )
        if isinstance(add_buy_distribution, dict) and add_buy_distribution:
            add_buy_summary += _format_add_buy_distribution(add_buy_distribution)
        if isinstance(add_buy_win_rates, dict) and add_buy_win_rates:
            add_buy_summary += _format_add_buy_win_rates(add_buy_win_rates)
        if isinstance(add_buy_exit_reasons, dict) and add_buy_exit_reasons:
            add_buy_summary += _format_add_buy_exit_reasons(add_buy_exit_reasons)
        if add_buy_debug_ready:
            add_buy_summary += _format_add_buy_debug(add_buy_debug)
    elif add_buy_debug_ready:
        add_buy_summary += _format_add_buy_debug(add_buy_debug)

    message = (
        "📊 Backtest Summary\n"
        f"🧠 Strategy: {strat_name}\n"
        f"🧾 Data: {data_symbol} | {data_interval} | {data_period}\n"
        f"🗓️ Backtest period: {period}\n"
        "----------------------------------------\n"
        f"🎯 Trades: {stats['total_trades']} | Wins: {stats['num_wins']} | "
        f"Losses: {stats['num_losses']} | Win rate: {win_rate:.2f}%\n"
        f"📈 Total return: {total_return:.2f}% | Max DD: {max_drawdown:.2f}% | "
        f"Profit factor: {profit_factor_text}\n"
        "----------------------------------------\n"
        f"📌 Avg trade: {stats['avg_trade']:.4f}\n"
        f"✅ Avg win: {stats['avg_win']:.4f} | ❌ Avg loss: {stats['avg_loss']:.4f}\n"
        f"📊 Avg return: {avg_return_pct:.2f}%\n"
        f"↕️ Avg min/max return: {avg_min_return_pct:.2f}% / {avg_max_return_pct:.2f}%\n"
        f"⚖️ Long/Short ratio: {long_ratio_pct:.2f}% / {short_ratio_pct:.2f}%\n"
        f"🟢 Long avg return: {long_avg_return_pct:.2f}% | 🔴 Short avg return: {short_avg_return_pct:.2f}%\n"
        "----------------------------------------\n"
        f"🏆 Largest win: {stats['largest_win']:.4f}\n"
        f"💥 Largest loss: {stats['largest_loss']:.4f}\n"
        f"🔥 Max win streak: {max_win_streak} | 💧 Max loss streak: {max_loss_streak}\n"
        f"{exit_reason_summary}"
        f"{add_buy_summary}"
        f"💰 Final equity: {stats['final_equity']:.2f}"
    )
    chart_paths: List[Path] = []
    chart_cfg = bt_cfg.get("chart", {})
    if isinstance(chart_cfg, dict):
        chart_enabled = chart_cfg.get("enabled", True)
    else:
        chart_enabled = True

    if chart_enabled:
        with tempfile.NamedTemporaryFile(prefix="backtest_", suffix=".png", delete=False) as tmp_file:
            equity_chart = create_backtest_chart(result, Path(tmp_file.name))
            if equity_chart:
                chart_paths.append(equity_chart)
        with tempfile.NamedTemporaryFile(prefix="backtest_stats_", suffix=".png", delete=False) as tmp_file:
            stats_chart = create_stats_chart(result, Path(tmp_file.name))
            if stats_chart:
                chart_paths.append(stats_chart)
        with tempfile.NamedTemporaryFile(prefix="backtest_trade_returns_", suffix=".png", delete=False) as tmp_file:
            trade_return_chart = create_trade_return_chart(result, Path(tmp_file.name))
            if trade_return_chart:
                chart_paths.append(trade_return_chart)
        with tempfile.NamedTemporaryFile(
            prefix="backtest_trade_distribution_", suffix=".png", delete=False
        ) as tmp_file:
            trade_distribution_chart = create_trade_distribution_chart(result, Path(tmp_file.name))
            if trade_distribution_chart:
                chart_paths.append(trade_distribution_chart)
        with tempfile.NamedTemporaryFile(prefix="backtest_holding_time_", suffix=".png", delete=False) as tmp_file:
            holding_time_chart = create_holding_time_chart(result, Path(tmp_file.name))
            if holding_time_chart:
                chart_paths.append(holding_time_chart)
        with tempfile.NamedTemporaryFile(prefix="backtest_daily_heatmap_", suffix=".png", delete=False) as tmp_file:
            daily_heatmaps = create_daily_return_heatmaps(result, Path(tmp_file.name))
            if daily_heatmaps:
                chart_paths.extend(daily_heatmaps)

    trades_path = _write_trade_log(result)
    if trades_path:
        chart_paths.append(trades_path)

    settings_path = _write_settings_snapshot(
        data_cfg=data_cfg,
        backtest_cfg=bt_cfg,
        strat_name=strat_name,
        strat_params=strat_params,
    )
    if settings_path:
        chart_paths.append(settings_path)

    return message, total_return, chart_paths, result, data


def run_backtest_from_settings(
    settings: Settings,
    data_overrides: Optional[Dict[str, Any]] = None,
    backtest_overrides: Optional[Dict[str, Any]] = None,
    strategy_overrides: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, float, List[Path]]:
    message, total_return, chart_paths, _, _ = _run_backtest(
        settings,
        data_overrides=data_overrides,
        backtest_overrides=backtest_overrides,
        strategy_overrides=strategy_overrides,
        progress_cb=progress_cb,
    )
    return message, total_return, chart_paths


def run_backtest_with_result(
    settings: Settings,
    data_overrides: Optional[Dict[str, Any]] = None,
    backtest_overrides: Optional[Dict[str, Any]] = None,
    strategy_overrides: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, float, List[Path], BacktestResult, pd.DataFrame]:
    return _run_backtest(
        settings,
        data_overrides=data_overrides,
        backtest_overrides=backtest_overrides,
        strategy_overrides=strategy_overrides,
        progress_cb=progress_cb,
    )


def run_backtest_with_entry_diff_result(
    settings: Settings,
    data_overrides: Optional[Dict[str, Any]] = None,
    backtest_overrides: Optional[Dict[str, Any]] = None,
    strategy_overrides: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    diff_config: Optional[EntryDiffConfig] = None,
) -> Tuple[str, float, List[Path], BacktestResult, pd.DataFrame]:
    message, total_return, chart_paths, result, data = _run_backtest(
        settings,
        data_overrides=data_overrides,
        backtest_overrides=backtest_overrides,
        strategy_overrides=strategy_overrides,
        progress_cb=progress_cb,
    )
    message = _truncate_backtest_summary(message)
    diff_message, diff_charts = create_entry_diff_report(
        result=result,
        data=data,
        config=diff_config,
    )
    if diff_message:
        message = f"{message}\n----------------------------------------\n{diff_message}"
    filtered_paths: List[Path] = [
        path for path in chart_paths if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}
    ]
    with tempfile.NamedTemporaryFile(prefix="backtest_equity_price_", suffix=".png", delete=False) as tmp_file:
        equity_price_chart = create_equity_price_chart(result, data, Path(tmp_file.name))
    if equity_price_chart:
        filtered_paths.append(equity_price_chart)
    if diff_charts:
        filtered_paths.extend(diff_charts)
    return message, total_return, filtered_paths, result, data


def _truncate_backtest_summary(message: str) -> str:
    if not message:
        return message
    lines = message.splitlines()
    for idx, line in enumerate(lines):
        if "Total return" in line:
            return "\n".join(lines[: idx + 1])
    return message


def run_backtest_summary(
    settings: Settings,
    data_overrides: Optional[Dict[str, Any]] = None,
    strategy_name: Optional[str] = None,
) -> Dict[str, Any]:
    data_cfg = _merge_config(settings.data, data_overrides)
    data_folder = data_cfg.get("folder", "price data")
    data_pattern = data_cfg.get("file_pattern", "*.csv")
    data = load_price_data(
        folder=data_folder,
        file_pattern=data_pattern,
        timestamp_column=data_cfg.get("timestamp_column", "timestamp"),
        timezone=data_cfg.get("timezone", "UTC"),
        datetime_format=data_cfg.get("datetime_format"),
    )

    strat_cfg = settings.strategy
    strat_name = strategy_name or strat_cfg.get("name", "sma_cross")
    strat_params: Dict[str, Any] = {}
    if strategy_name is None or strat_name == strat_cfg.get("name"):
        strat_params = dict(strat_cfg.get("params", {}))
    if str(strat_name).strip().lower().replace(" ", "_").replace("-", "_") == "cheatkey":
        strat_params = _normalize_cheatkey_params(strat_params)
    strategy_cls = get_strategy_class(strat_name)
    try:
        strategy = strategy_cls(**strat_params)
    except TypeError:
        strategy = strategy_cls()

    bt_cfg = settings.backtest
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_balance=bt_cfg.get("initial_balance", 10000.0),
        fee_rate=bt_cfg.get("fee_rate", 0.0004),
        leverage=bt_cfg.get("leverage", 1.0),
        position_size=bt_cfg.get("position_size", 1.0),
        max_position_fraction_per_side=bt_cfg.get("max_position_fraction_per_side"),
        risk_per_trade=bt_cfg.get("risk_per_trade"),
        slippage=bt_cfg.get("slippage", 0.0),
    )

    result = engine.run()
    stats = result.stats()
    stats["strategy_name"] = strat_name
    return stats


def run_backtest_from_path(settings_path: str | Path) -> Tuple[str, float, List[Path]]:
    settings = load_settings(Path(settings_path))
    return run_backtest_from_settings(settings)
