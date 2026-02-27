from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any

import pandas as pd

from backtest.result import BacktestResult
from backtest.strategy import Strategy


@dataclass
class Position:
    direction: int
    entry_price: float
    qty: float
    entry_time: pd.Timestamp
    entry_index: int
    leverage: float = 1.0
    min_return: float = 0.0
    max_return: float = 0.0
    entry_reason: str = ""
    stop_price: float | None = None
    prev_extreme_stop_price: float | None = None
    rr_stop_price: float | None = None
    rr_tp_price: float | None = None
    moved_sl_on_profit: bool = False
    moved_tp_on_loss: bool = False
    risk_pct: float = 0.0
    base_qty: float = 0.0
    base_cost: float = 0.0
    added_qty: float = 0.0
    added_cost: float = 0.0
    part_qty: float = 0.0
    base_parts: int = 1
    target_qty: float = 0.0
    parts_filled: int = 0
    last_signal_index: int = 0
    slope_exit_cooldown: int = 0


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


class BacktestEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0004,
        leverage: float = 1.0,
        position_size: float = 1.0,
        max_position_fraction_per_side: float | None = None,
        risk_per_trade: float | None = None,
        slippage: float = 0.0,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.position_size = position_size
        self.max_position_fraction_per_side = max_position_fraction_per_side
        self.risk_per_trade = risk_per_trade
        self.slippage = slippage

    def run(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        close_open_positions: bool = True,
        ignore_last_row_signals: bool = False,
        suppress_signals_from_index: Optional[int] = None,
    ) -> BacktestResult:
        if self.data.empty:
            raise ValueError("No data to backtest")
        if "open" not in self.data.columns or "close" not in self.data.columns:
            raise ValueError("Data must contain 'open' and 'close' columns")

        total_rows = len(self.data)
        if progress_cb:
            progress_cb(0, total_rows)
            progress_step = max(1, total_rows // 100)
        else:
            progress_step = 0

        signals = self.strategy.generate_signals(self.data)
        signals = signals.reindex(self.data.index).fillna(0).astype(int)
        reasons = self.strategy.signal_reasons(self.data)
        if reasons is None:
            reasons = pd.Series([""] * len(self.data), index=self.data.index)
        reasons = reasons.reindex(self.data.index).fillna("").astype(str)
        exit_signals = self.strategy.exit_signals(self.data)
        if exit_signals is None:
            exit_signals = pd.Series([0] * len(self.data), index=self.data.index)
        exit_signals = exit_signals.reindex(self.data.index).fillna(0).astype(int)
        if ignore_last_row_signals and len(self.data) > 0:
            signals = signals.copy()
            reasons = reasons.copy()
            exit_signals = exit_signals.copy()
            signals.iloc[-1] = 0
            exit_signals.iloc[-1] = 0
            reasons.iloc[-1] = ""
        if suppress_signals_from_index is not None and len(self.data) > 0:
            idx = int(suppress_signals_from_index)
            if idx < 0:
                idx = 0
            if idx < len(self.data):
                signals = signals.copy()
                reasons = reasons.copy()
                exit_signals = exit_signals.copy()
                signals.iloc[idx:] = 0
                exit_signals.iloc[idx:] = 0
                reasons.iloc[idx:] = ""

        cash = float(self.initial_balance)
        long_position: Optional[Position] = None
        short_position: Optional[Position] = None
        trades: List[Dict[str, float]] = []
        equity_rows = []

        stop_atr_mult = getattr(self.strategy, "stop_atr_mult", None)
        swing_lookback = getattr(self.strategy, "swing_lookback", None)
        atr_window = getattr(self.strategy, "atr_window", None)
        time_stop_bars = getattr(self.strategy, "time_stop_bars", None)
        time_stop_min_r = getattr(self.strategy, "time_stop_min_r", None)
        long_leverage = getattr(self.strategy, "long_leverage", None)
        short_leverage = getattr(self.strategy, "short_leverage", None)
        long_tp_pnl = getattr(self.strategy, "long_tp_pnl", None)
        short_tp_pnl = getattr(self.strategy, "short_tp_pnl", None)
        long_sl_pnl = getattr(self.strategy, "long_sl_pnl", None)
        short_sl_pnl = getattr(self.strategy, "short_sl_pnl", None)
        long_add_sl_pnl = getattr(self.strategy, "long_add_sl_pnl", None)
        short_add_sl_pnl = getattr(self.strategy, "short_add_sl_pnl", None)
        long_add_tp_pnl = getattr(self.strategy, "long_add_tp_pnl", None)
        short_add_tp_pnl = getattr(self.strategy, "short_add_tp_pnl", None)
        use_prev_extreme_stop = getattr(self.strategy, "use_prev_extreme_stop", False)
        prev_extreme_lookback = getattr(self.strategy, "prev_extreme_lookback", None)
        prev_extreme_buffer_pct = getattr(self.strategy, "prev_extreme_buffer_pct", None)
        prev_extreme_max_loss_pct = getattr(self.strategy, "prev_extreme_max_loss_pct", None)
        use_prev_extreme_rr = getattr(self.strategy, "use_prev_extreme_rr", False)
        prev_extreme_rr_lookback = getattr(self.strategy, "prev_extreme_rr_lookback", None)
        prev_extreme_rr_buffer_pct = getattr(self.strategy, "prev_extreme_rr_buffer_pct", None)
        prev_extreme_rr_multiplier = getattr(self.strategy, "prev_extreme_rr_multiplier", None)
        prev_extreme_rr_max_loss_pct = getattr(self.strategy, "prev_extreme_rr_max_loss_pct", None)
        prev_extreme_rr_min_loss_pct = getattr(self.strategy, "prev_extreme_rr_min_loss_pct", None)
        use_prev_extreme_rr_tp_cap = getattr(self.strategy, "use_prev_extreme_rr_tp_cap", False)
        prev_extreme_rr_tp_cap_lookback = getattr(
            self.strategy, "prev_extreme_rr_tp_cap_lookback", None
        )
        prev_extreme_rr_tp_cap_buffer_pct = getattr(
            self.strategy, "prev_extreme_rr_tp_cap_buffer_pct", None
        )
        move_sl_on_profit = getattr(self.strategy, "move_sl_on_profit", False)
        move_sl_trigger_pnl = getattr(self.strategy, "move_sl_trigger_pnl", None)
        move_sl_target_pnl = getattr(self.strategy, "move_sl_target_pnl", None)
        move_sl_use_ratio = getattr(self.strategy, "move_sl_use_ratio", False)
        move_sl_trigger_ratio = getattr(self.strategy, "move_sl_trigger_ratio", None)
        move_sl_target_ratio = getattr(self.strategy, "move_sl_target_ratio", None)
        move_tp_on_loss = getattr(self.strategy, "move_tp_on_loss", False)
        move_tp_trigger_pnl = getattr(self.strategy, "move_tp_trigger_pnl", None)
        move_tp_target_pnl = getattr(self.strategy, "move_tp_target_pnl", None)
        use_rr_tp_from_sl_pnl = getattr(self.strategy, "use_rr_tp_from_sl_pnl", False)
        rr_tp_from_sl_multiplier = getattr(self.strategy, "rr_tp_from_sl_multiplier", None)
        auto_leverage_min_loss_pnl = getattr(self.strategy, "auto_leverage_min_loss_pnl", None)
        auto_leverage_max_loss_pnl = getattr(self.strategy, "auto_leverage_max_loss_pnl", None)
        auto_leverage_max_leverage = getattr(self.strategy, "auto_leverage_max_leverage", None)
        long_partial_exit_pnl = getattr(self.strategy, "long_partial_exit_pnl", None)
        short_partial_exit_pnl = getattr(self.strategy, "short_partial_exit_pnl", None)
        partial_exit_mode = getattr(self.strategy, "partial_exit_mode", None)
        long_add_buy_pnl = getattr(self.strategy, "long_add_buy_pnl", None)
        short_add_buy_pnl = getattr(self.strategy, "short_add_buy_pnl", None)
        add_buy_candle_condition = getattr(self.strategy, "add_buy_candle_condition", False)
        use_add_buy = getattr(self.strategy, "use_add_buy", True)
        add_buy_max_times = getattr(self.strategy, "add_buy_max_times", None)
        add_buy_use_rr_ratio = getattr(self.strategy, "add_buy_use_rr_ratio", False)
        add_buy_rr_ratio = getattr(self.strategy, "add_buy_rr_ratio", None)
        divide_value = getattr(self.strategy, "divide_value", None)
        total_parts = getattr(self.strategy, "total_parts", None)
        exit_mode = getattr(self.strategy, "exit_mode", None)
        exit_full_condition = getattr(self.strategy, "exit_full_condition", "always")
        first_opposite_exit = getattr(self.strategy, "first_opposite_exit", False)
        first_opposite_exit_pnl = getattr(self.strategy, "first_opposite_exit_pnl", 0.0)
        exitmode_timefilter = getattr(self.strategy, "exitmode_timefilter", False)
        use_timeout_cross = getattr(self.strategy, "use_timeout_cross", False)
        timeout_cross_bars = getattr(self.strategy, "timeout_cross_bars", None)
        timeout_sell_pnl = getattr(self.strategy, "timeout_sell_pnl", None)
        use_second_buy = getattr(self.strategy, "use_second_buy", False)
        use_first_add_macro_ema = getattr(self.strategy, "use_first_add_macro_ema", False)
        first_add_candle_condition_enabled = getattr(
            self.strategy, "first_add_candle_condition_enabled", False
        )
        use_4h_candle_condition = getattr(self.strategy, "use_4h_candle_condition", False)
        bb_width_window = getattr(self.strategy, "bb_width_window", 20)
        bb_width_std = getattr(self.strategy, "bb_width_std", 2.0)
        bb_width_threshold = getattr(self.strategy, "bb_width_threshold", 0.05)
        use_override_atr = getattr(self.strategy, "use_override_atr", False)
        atr_override_period = getattr(self.strategy, "atr_override_period", None)
        atr_pct_threshold = getattr(self.strategy, "atr_pct_threshold", None)
        unit_override = getattr(self.strategy, "unit_override", 1.0)
        macro_ema_filter = getattr(self.strategy, "macro_ema_filter", False)
        macro_ema_window = getattr(self.strategy, "macro_ema_window", None)
        slope_exit_mode = getattr(self.strategy, "slope_exit_mode", None)
        slope_exit_cooldown_bars = getattr(self.strategy, "slope_exit_cooldown_bars", None)
        slope_exit_pnl_threshold = getattr(self.strategy, "slope_exit_pnl_threshold", None)
        slope_exit_use_threshold = getattr(self.strategy, "slope_exit_use_threshold", True)
        slope_exit_ignore_threshold_after_add = getattr(
            self.strategy, "slope_exit_ignore_threshold_after_add", False
        )

        use_stop = (
            stop_atr_mult is not None
            and swing_lookback is not None
            and atr_window is not None
            and {"high", "low", "close"}.issubset(self.data.columns)
        )
        if use_stop:
            atr = _atr(self.data, int(atr_window))
            swing_high = self.data["high"].rolling(int(swing_lookback), min_periods=int(swing_lookback)).max()
            swing_low = self.data["low"].rolling(int(swing_lookback), min_periods=int(swing_lookback)).min()
        else:
            atr = None
            swing_high = None
            swing_low = None

        atr_override_factor = None
        if (
            use_override_atr
            and atr_override_period is not None
            and atr_pct_threshold is not None
            and {"high", "low", "close"}.issubset(self.data.columns)
        ):
            atr_override = _atr(self.data, int(atr_override_period))
            close = self.data["close"].replace(0, pd.NA)
            atr_pct = (atr_override / close) * 100.0
            atr_override_factor = pd.Series(1.0, index=self.data.index, dtype=float)
            atr_override_factor[atr_pct <= float(atr_pct_threshold)] = float(unit_override)

        bb_width_narrow = None
        if use_4h_candle_condition and "timestamp" in self.data.columns:
            ts = pd.to_datetime(self.data["timestamp"])
            close = self.data["close"].astype(float)
            close_4h = (
                pd.DataFrame({"close": close.values}, index=ts)
                .resample("4h")
                .last()["close"]
            )
            ma = close_4h.rolling(int(bb_width_window), min_periods=int(bb_width_window)).mean()
            std = close_4h.rolling(int(bb_width_window), min_periods=int(bb_width_window)).std()
            upper = ma + float(bb_width_std) * std
            lower = ma - float(bb_width_std) * std
            width = (upper - lower) / ma.replace(0, pd.NA)
            narrow_4h = width <= float(bb_width_threshold)
            bb_width_narrow = narrow_4h.reindex(ts, method="ffill").fillna(False)

        ema_cross_signals = None
        if use_second_buy:
            if hasattr(self.strategy, "ema_cross_signals"):
                ema_cross_signals = self.strategy.ema_cross_signals(self.data)
            else:
                fast = getattr(self.strategy, "ema_fast", None)
                slow = getattr(self.strategy, "ema_slow", None)
                if fast is not None and slow is not None:
                    ema_fast = _ema(self.data["close"], int(fast))
                    ema_slow = _ema(self.data["close"], int(slow))
                    prev_fast = ema_fast.shift(1)
                    prev_slow = ema_slow.shift(1)
                    cross_up = (ema_fast > ema_slow) & (prev_fast <= prev_slow)
                    cross_down = (ema_fast < ema_slow) & (prev_fast >= prev_slow)
                    ema_cross_signals = pd.Series(0, index=self.data.index, dtype=int)
                    ema_cross_signals[cross_up] = 1
                    ema_cross_signals[cross_down] = -1

        macro_ema = None
        macro_long_prev = None
        macro_short_prev = None
        if macro_ema_filter and macro_ema_window:
            macro_ema = _ema(self.data["close"], int(macro_ema_window))
            current_close = self.data["close"]
            macro_long_prev = current_close > macro_ema
            macro_short_prev = current_close < macro_ema

        def _update_position_extremes(
            position: Position,
            close_price: float,
            high_price: float,
            low_price: float,
        ) -> float:
            equity_delta = (close_price - position.entry_price) * position.qty * position.direction
            entry_price = position.entry_price
            if entry_price > 0:
                leverage_value = position.leverage if position.leverage > 0 else 1.0
                if position.direction > 0:
                    max_ret = (high_price - entry_price) / entry_price
                    min_ret = (low_price - entry_price) / entry_price
                else:
                    max_ret = (entry_price - low_price) / entry_price
                    min_ret = (entry_price - high_price) / entry_price
                position.max_return = max(position.max_return, max_ret * leverage_value)
                position.min_return = min(position.min_return, min_ret * leverage_value)
            return equity_delta

        def _resolve_parts() -> int:
            parts = int(total_parts) if total_parts else (2 ** int(divide_value) if divide_value else 1)
            if parts <= 0:
                parts = 1
            return parts

        def _resolve_max_additions() -> int | None:
            if not use_add_buy or add_buy_max_times is None:
                return None
            try:
                max_additions = int(add_buy_max_times)
            except (TypeError, ValueError):
                return None
            return max(0, max_additions)

        def _target_price_for_pnl(
            entry_price: float,
            direction: int,
            leverage_value: float,
            pnl_target: float,
        ) -> float | None:
            if entry_price <= 0:
                return None
            if leverage_value <= 0:
                leverage_value = 1.0
            pnl_unlevered = float(pnl_target) / leverage_value
            if direction > 0:
                return entry_price * (1.0 + pnl_unlevered)
            return entry_price * (1.0 - pnl_unlevered)

        def _resolve_effective_tp_sl(position: Position, index: int) -> tuple[float | None, float | None]:
            tp_pnl = long_tp_pnl if position.direction > 0 else short_tp_pnl
            sl_pnl = long_sl_pnl if position.direction > 0 else short_sl_pnl
            add_sl_list = long_add_sl_pnl if position.direction > 0 else short_add_sl_pnl
            add_tp_list = long_add_tp_pnl if position.direction > 0 else short_add_tp_pnl
            override_factor = 1.0
            if atr_override_factor is not None:
                override_factor = float(atr_override_factor.iloc[index])
            if tp_pnl is not None:
                tp_pnl = float(tp_pnl) * override_factor
            if sl_pnl is not None:
                sl_pnl = float(sl_pnl) * override_factor
            if add_sl_list:
                add_sl_list = [float(v) * override_factor for v in add_sl_list]
            if add_tp_list:
                add_tp_list = [float(v) * override_factor for v in add_tp_list]
            if position.parts_filled > 1 and add_sl_list:
                idx = min(position.parts_filled - 2, len(add_sl_list) - 1)
                sl_pnl = add_sl_list[idx]
            if position.parts_filled > 1 and add_tp_list:
                idx = min(position.parts_filled - 2, len(add_tp_list) - 1)
                tp_pnl = add_tp_list[idx]
            if use_rr_tp_from_sl_pnl:
                try:
                    rr_mult = float(rr_tp_from_sl_multiplier)
                except (TypeError, ValueError):
                    rr_mult = None
                if rr_mult is not None and rr_mult > 0 and sl_pnl is not None:
                    try:
                        sl_val = float(sl_pnl)
                    except (TypeError, ValueError):
                        sl_val = None
                    if sl_val is not None:
                        tp_pnl = abs(sl_val) * rr_mult
            return tp_pnl, sl_pnl

        def _enter_position(
            signal: int,
            next_open: float,
            entry_index: int,
            entry_time: pd.Timestamp,
            entry_reason: str,
            atr_index: int,
        ) -> Optional[Position]:
            nonlocal cash
            leverage_override = long_leverage if signal > 0 else short_leverage
            leverage_value = leverage_override if leverage_override is not None else self.leverage
            if leverage_value is None or leverage_value <= 0:
                leverage_value = 1.0
            stop_price = None
            prev_extreme_stop_price = None
            rr_stop_price = None
            rr_tp_price = None
            stop_distance = None
            risk_pct = 0.0
            rr_min_loss_target = None

            def _cap_prev_extreme_stop(
                stop_price_val: float | None,
                max_loss_pct: float | None,
            ) -> float | None:
                if stop_price_val is None:
                    return None
                if max_loss_pct is None:
                    return stop_price_val
                try:
                    max_loss_val = abs(float(max_loss_pct))
                except (TypeError, ValueError):
                    return stop_price_val
                if max_loss_val <= 0:
                    return stop_price_val
                max_unlevered = max_loss_val / leverage_value
                if max_unlevered <= 0:
                    return stop_price_val
                if signal > 0:
                    cap_price = float(next_open) * (1.0 - max_unlevered)
                    if cap_price > 0 and stop_price_val < cap_price:
                        return cap_price
                else:
                    cap_price = float(next_open) * (1.0 + max_unlevered)
                    if cap_price > 0 and stop_price_val > cap_price:
                        return cap_price
                return stop_price_val
            if use_stop and atr is not None and swing_high is not None and swing_low is not None:
                atr_val = atr.iloc[atr_index]
                if pd.notna(atr_val):
                    if signal > 0:
                        swing_low_val = swing_low.iloc[atr_index]
                        if pd.notna(swing_low_val):
                            stop_price = float(swing_low_val) - float(stop_atr_mult) * float(atr_val)
                    else:
                        swing_high_val = swing_high.iloc[atr_index]
                        if pd.notna(swing_high_val):
                            stop_price = float(swing_high_val) + float(stop_atr_mult) * float(atr_val)
                if stop_price is not None and stop_price > 0:
                    stop_distance = abs(next_open - stop_price)
                    if stop_distance > 0:
                        risk_pct = stop_distance / next_open

            if use_prev_extreme_rr and prev_extreme_rr_lookback:
                try:
                    lookback_n = max(int(prev_extreme_rr_lookback), 1)
                except (TypeError, ValueError):
                    lookback_n = 0
                if lookback_n > 0 and entry_index - lookback_n >= 0:
                    if "low" in self.data.columns and "high" in self.data.columns:
                        window = self.data.iloc[entry_index - lookback_n : entry_index]
                        buffer_pct = float(prev_extreme_rr_buffer_pct or 0.0)
                        if signal > 0:
                            prior_low = window["low"].min()
                            if pd.notna(prior_low) and float(prior_low) > 0:
                                rr_stop_price = float(prior_low) * (1.0 - buffer_pct)
                        else:
                            prior_high = window["high"].max()
                            if pd.notna(prior_high) and float(prior_high) > 0:
                                rr_stop_price = float(prior_high) * (1.0 + buffer_pct)
                        if rr_stop_price is not None and float(next_open) > 0:
                            if signal > 0 and rr_stop_price >= float(next_open):
                                rr_stop_price = None
                            elif signal < 0 and rr_stop_price <= float(next_open):
                                rr_stop_price = None
                        if rr_stop_price is not None:
                            rr_max_loss = (
                                prev_extreme_rr_max_loss_pct
                                if prev_extreme_rr_max_loss_pct is not None
                                else prev_extreme_max_loss_pct
                            )
                            rr_stop_price = _cap_prev_extreme_stop(
                                rr_stop_price,
                                rr_max_loss,
                            )
                            rr_min_loss = prev_extreme_rr_min_loss_pct
                            if rr_min_loss is not None and rr_stop_price is not None:
                                try:
                                    rr_min_loss_val = abs(float(rr_min_loss))
                                except (TypeError, ValueError):
                                    rr_min_loss_val = None
                                if rr_min_loss_val and rr_min_loss_val > 0:
                                    risk_distance = abs(float(next_open) - float(rr_stop_price))
                                    if float(next_open) > 0 and risk_distance > 0:
                                        leveraged_loss = (risk_distance / float(next_open)) * leverage_value
                                        if leveraged_loss < rr_min_loss_val:
                                            rr_min_loss_target = rr_min_loss_val
                                            if auto_leverage_min_loss_pnl is None:
                                                return None
                            rr_mult = None
                            if use_rr_tp_from_sl_pnl:
                                try:
                                    rr_mult = float(rr_tp_from_sl_multiplier)
                                except (TypeError, ValueError):
                                    rr_mult = None
                            if rr_mult is None:
                                try:
                                    rr_mult = float(prev_extreme_rr_multiplier)
                                except (TypeError, ValueError):
                                    rr_mult = None
                            if rr_mult is not None and rr_mult > 0:
                                risk_distance = abs(float(next_open) - float(rr_stop_price))
                                if risk_distance > 0:
                                    if signal > 0:
                                        rr_tp_price = float(next_open) + risk_distance * rr_mult
                                    else:
                                        rr_tp_price = float(next_open) - risk_distance * rr_mult
                            if (
                                rr_tp_price is not None
                                and use_prev_extreme_rr_tp_cap
                                and prev_extreme_rr_tp_cap_lookback
                            ):
                                try:
                                    tp_lookback_n = max(int(prev_extreme_rr_tp_cap_lookback), 1)
                                except (TypeError, ValueError):
                                    tp_lookback_n = 0
                                if tp_lookback_n > 0 and entry_index - tp_lookback_n >= 0:
                                    if "low" in self.data.columns and "high" in self.data.columns:
                                        tp_window = self.data.iloc[
                                            entry_index - tp_lookback_n : entry_index
                                        ]
                                        tp_buffer = float(prev_extreme_rr_tp_cap_buffer_pct or 0.0)
                                        if signal > 0:
                                            prior_high = tp_window["high"].max()
                                            if pd.notna(prior_high) and float(prior_high) > 0:
                                                if rr_tp_price > float(prior_high):
                                                    cap_price = float(prior_high) * (1.0 - tp_buffer)
                                                    if cap_price > float(next_open):
                                                        rr_tp_price = cap_price
                                        else:
                                            prior_low = tp_window["low"].min()
                                            if pd.notna(prior_low) and float(prior_low) > 0:
                                                if rr_tp_price < float(prior_low):
                                                    cap_price = float(prior_low) * (1.0 + tp_buffer)
                                                    if cap_price < float(next_open):
                                                        rr_tp_price = cap_price

            if (not use_prev_extreme_rr) and use_prev_extreme_stop and prev_extreme_lookback:
                try:
                    lookback_n = max(int(prev_extreme_lookback), 1)
                except (TypeError, ValueError):
                    lookback_n = 0
                if lookback_n > 0 and entry_index - lookback_n >= 0:
                    if "low" in self.data.columns and "high" in self.data.columns:
                        window = self.data.iloc[entry_index - lookback_n : entry_index]
                        buffer_pct = float(prev_extreme_buffer_pct or 0.0)
                        if signal > 0:
                            prior_low = window["low"].min()
                            if pd.notna(prior_low) and float(prior_low) > 0:
                                prev_extreme_stop_price = float(prior_low) * (1.0 - buffer_pct)
                        else:
                            prior_high = window["high"].max()
                            if pd.notna(prior_high) and float(prior_high) > 0:
                                prev_extreme_stop_price = float(prior_high) * (1.0 + buffer_pct)
                        if prev_extreme_stop_price is not None:
                            prev_extreme_stop_price = _cap_prev_extreme_stop(
                                prev_extreme_stop_price,
                                prev_extreme_max_loss_pct,
                            )

            auto_min_loss = None
            if auto_leverage_min_loss_pnl is not None:
                try:
                    auto_min_loss = abs(float(auto_leverage_min_loss_pnl))
                except (TypeError, ValueError):
                    auto_min_loss = None
            auto_max_loss = None
            if auto_leverage_max_loss_pnl is not None:
                try:
                    auto_max_loss = abs(float(auto_leverage_max_loss_pnl))
                except (TypeError, ValueError):
                    auto_max_loss = None
            leverage_min_target = None
            if rr_min_loss_target is not None and auto_min_loss is not None:
                leverage_min_target = max(rr_min_loss_target, auto_min_loss)
            elif rr_min_loss_target is not None:
                leverage_min_target = rr_min_loss_target
            else:
                leverage_min_target = auto_min_loss
            if (leverage_min_target is not None and leverage_min_target > 0) or (
                auto_max_loss is not None and auto_max_loss > 0
            ):
                stop_for_leverage = None
                if stop_price is not None:
                    stop_for_leverage = stop_price
                elif rr_stop_price is not None:
                    stop_for_leverage = rr_stop_price
                elif prev_extreme_stop_price is not None:
                    stop_for_leverage = prev_extreme_stop_price
                if stop_for_leverage is not None and float(next_open) > 0:
                    risk_distance = abs(float(next_open) - float(stop_for_leverage))
                    if risk_distance > 0:
                        risk_pct_for_leverage = risk_distance / float(next_open)
                        target_leverage = leverage_value
                        if leverage_min_target is not None and leverage_min_target > 0:
                            target_leverage = max(
                                target_leverage,
                                leverage_min_target / risk_pct_for_leverage,
                            )
                        if auto_max_loss is not None and auto_max_loss > 0:
                            target_leverage = min(
                                target_leverage,
                                auto_max_loss / risk_pct_for_leverage,
                            )
                        max_leverage = None
                        if auto_leverage_max_leverage is not None:
                            try:
                                max_leverage = float(auto_leverage_max_leverage)
                            except (TypeError, ValueError):
                                max_leverage = None
                        if max_leverage is not None and max_leverage > 0:
                            target_leverage = min(target_leverage, max_leverage)
                        if target_leverage > 0:
                            leverage_value = target_leverage

            qty = self._calc_qty(cash, next_open, stop_distance, leverage_value)
            parts = _resolve_parts()
            part_qty = qty / parts
            fee = abs(part_qty * next_open) * self.fee_rate
            base_cost = part_qty * next_open
            required_cash = (base_cost / leverage_value) + fee if leverage_value > 0 else base_cost + fee
            if cash < required_cash:
                return None
            cash -= required_cash
            position = Position(
                signal,
                next_open,
                part_qty,
                entry_time,
                entry_index,
                entry_reason=entry_reason,
                stop_price=stop_price,
                prev_extreme_stop_price=prev_extreme_stop_price,
                rr_stop_price=rr_stop_price,
                rr_tp_price=rr_tp_price,
                risk_pct=risk_pct,
                leverage=leverage_value,
                base_qty=part_qty,
                base_cost=base_cost,
                added_qty=0.0,
                added_cost=0.0,
                part_qty=part_qty,
                base_parts=parts,
                target_qty=qty,
                parts_filled=1,
                last_signal_index=atr_index,
                slope_exit_cooldown=int(slope_exit_cooldown_bars)
                if slope_exit_cooldown_bars is not None
                else 0,
            )
            position.qty = position.base_qty + position.added_qty
            if position.qty > 0:
                position.entry_price = (position.base_cost + position.added_cost) / position.qty
            trades.append(
                {
                    "type": "entry",
                    "direction": signal,
                    "price": next_open,
                    "qty": part_qty,
                    "fee": fee,
                    "leverage": leverage_value,
                    "timestamp": entry_time,
                    "reason": entry_reason,
                    "stop_price": stop_price,
                    "rr_stop_price": rr_stop_price,
                    "rr_tp_price": rr_tp_price,
                }
            )
            return position

        def _process_exits(
            position: Optional[Position],
            i: int,
            row: pd.Series,
            timestamp: pd.Timestamp,
            close_price: float,
            high_price: float,
            low_price: float,
        ) -> Optional[Position]:
            nonlocal cash
            if position is None:
                return None

            entry_price = position.entry_price
            leverage_value = position.leverage if position.leverage > 0 else 1.0

            if use_stop:
                stop_hit = False
                stop_price = position.stop_price
                if stop_price is not None:
                    if position.direction > 0 and low_price <= stop_price:
                        stop_hit = True
                    elif position.direction < 0 and high_price >= stop_price:
                        stop_hit = True
                if stop_hit:
                    cash = self._close_position(
                        cash=cash,
                        position=position,
                        trades=trades,
                        exit_price=stop_price if stop_price is not None else close_price,
                        timestamp=timestamp,
                        exit_reason="stop_loss_atr",
                    )
                    return None

            if use_prev_extreme_rr:
                rr_stop_price = position.rr_stop_price
                if move_sl_on_profit:
                    trigger_pnl = None
                    target_pnl = None
                    if move_sl_use_ratio:
                        tp_pnl, sl_pnl = _resolve_effective_tp_sl(position, i)
                        try:
                            trigger_ratio = float(move_sl_trigger_ratio)
                            target_ratio = float(move_sl_target_ratio)
                        except (TypeError, ValueError):
                            trigger_ratio = None
                            target_ratio = None
                        if (
                            trigger_ratio is not None
                            and target_ratio is not None
                            and tp_pnl is not None
                            and sl_pnl is not None
                        ):
                            trigger_pnl = float(tp_pnl) * trigger_ratio
                            target_pnl = float(sl_pnl) * target_ratio
                    elif move_sl_trigger_pnl is not None and move_sl_target_pnl is not None:
                        try:
                            trigger_pnl = float(move_sl_trigger_pnl)
                            target_pnl = float(move_sl_target_pnl)
                        except (TypeError, ValueError):
                            trigger_pnl = None
                            target_pnl = None
                    try:
                        trigger_pnl = float(trigger_pnl) if trigger_pnl is not None else None
                        target_pnl = float(target_pnl) if target_pnl is not None else None
                    except (TypeError, ValueError):
                        trigger_pnl = None
                        target_pnl = None
                    if (
                        trigger_pnl is not None
                        and target_pnl is not None
                        and trigger_pnl > 0
                        and position.max_return >= trigger_pnl
                        and position.max_return >= target_pnl
                    ):
                        prev_rr_stop_price = rr_stop_price
                        target_price = None
                        if entry_price > 0:
                            target_unlevered = float(target_pnl) / leverage_value
                            if position.direction > 0:
                                target_price = entry_price * (1.0 + target_unlevered)
                            else:
                                target_price = entry_price * (1.0 + abs(target_unlevered))
                        if target_price is not None and target_price > 0:
                            if rr_stop_price is None:
                                rr_stop_price = target_price
                            elif position.direction > 0:
                                rr_stop_price = max(rr_stop_price, target_price)
                            else:
                                rr_stop_price = min(rr_stop_price, target_price)
                            position.rr_stop_price = rr_stop_price
                        if prev_rr_stop_price != rr_stop_price and rr_stop_price is not None:
                            position.moved_sl_on_profit = True
                if rr_stop_price is not None:
                    if position.direction > 0 and low_price <= rr_stop_price:
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=rr_stop_price,
                            timestamp=timestamp,
                            exit_reason=(
                                "prev_extreme_rr_stop_moved"
                                if position.moved_sl_on_profit
                                else "prev_extreme_rr_stop"
                            ),
                        )
                        return None
                    if position.direction < 0 and high_price >= rr_stop_price:
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=rr_stop_price,
                            timestamp=timestamp,
                            exit_reason=(
                                "prev_extreme_rr_stop_moved"
                                if position.moved_sl_on_profit
                                else "prev_extreme_rr_stop"
                            ),
                        )
                        return None
                rr_tp_price = position.rr_tp_price
                if move_tp_on_loss and move_tp_trigger_pnl is not None and move_tp_target_pnl is not None:
                    try:
                        trigger_pnl = float(move_tp_trigger_pnl)
                        target_pnl = float(move_tp_target_pnl)
                    except (TypeError, ValueError):
                        trigger_pnl = None
                        target_pnl = None
                    if (
                        trigger_pnl is not None
                        and target_pnl is not None
                        and trigger_pnl < 0
                        and target_pnl > 0
                        and position.min_return <= trigger_pnl
                    ):
                        prev_rr_tp_price = rr_tp_price
                        target_price = None
                        if entry_price > 0:
                            target_unlevered = float(target_pnl) / leverage_value
                            if position.direction > 0:
                                target_price = entry_price * (1.0 + target_unlevered)
                            else:
                                target_price = entry_price * (1.0 - target_unlevered)
                        if target_price is not None and target_price > 0:
                            if rr_tp_price is None:
                                rr_tp_price = target_price
                            elif position.direction > 0:
                                rr_tp_price = max(rr_tp_price, target_price)
                            else:
                                rr_tp_price = min(rr_tp_price, target_price)
                            position.rr_tp_price = rr_tp_price
                        if prev_rr_tp_price != rr_tp_price and rr_tp_price is not None:
                            position.moved_tp_on_loss = True
                if rr_tp_price is not None:
                    if position.direction > 0 and high_price >= rr_tp_price:
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=rr_tp_price,
                            timestamp=timestamp,
                            exit_reason=(
                                "prev_extreme_rr_tp_moved"
                                if position.moved_tp_on_loss
                                else "prev_extreme_rr_tp"
                            ),
                        )
                        return None
                    if position.direction < 0 and low_price <= rr_tp_price:
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=rr_tp_price,
                            timestamp=timestamp,
                            exit_reason=(
                                "prev_extreme_rr_tp_moved"
                                if position.moved_tp_on_loss
                                else "prev_extreme_rr_tp"
                            ),
                        )
                        return None
            else:
                prev_extreme_stop_price = position.prev_extreme_stop_price
                if prev_extreme_stop_price is not None:
                    if position.direction > 0 and low_price <= prev_extreme_stop_price:
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=prev_extreme_stop_price,
                            timestamp=timestamp,
                            exit_reason="prev_extreme_stop",
                        )
                        return None
                    if position.direction < 0 and high_price >= prev_extreme_stop_price:
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=prev_extreme_stop_price,
                            timestamp=timestamp,
                            exit_reason="prev_extreme_stop",
                        )
                        return None

            if not use_prev_extreme_rr:
                tp_pnl, sl_pnl = _resolve_effective_tp_sl(position, i)

                if move_sl_on_profit:
                    trigger_pnl = None
                    target_pnl = None
                    if move_sl_use_ratio:
                        try:
                            trigger_ratio = float(move_sl_trigger_ratio)
                            target_ratio = float(move_sl_target_ratio)
                        except (TypeError, ValueError):
                            trigger_ratio = None
                            target_ratio = None
                        if (
                            trigger_ratio is not None
                            and target_ratio is not None
                            and tp_pnl is not None
                            and sl_pnl is not None
                        ):
                            trigger_pnl = float(tp_pnl) * trigger_ratio
                            target_pnl = float(sl_pnl) * target_ratio
                    elif move_sl_trigger_pnl is not None and move_sl_target_pnl is not None:
                        try:
                            trigger_pnl = float(move_sl_trigger_pnl)
                            target_pnl = float(move_sl_target_pnl)
                        except (TypeError, ValueError):
                            trigger_pnl = None
                            target_pnl = None
                    if (
                        trigger_pnl is not None
                        and target_pnl is not None
                        and trigger_pnl > 0
                        and position.max_return >= trigger_pnl
                        and position.max_return >= target_pnl
                    ):
                        prev_sl_pnl = sl_pnl
                        if sl_pnl is None:
                            sl_pnl = target_pnl
                        else:
                            sl_pnl = max(float(sl_pnl), target_pnl)
                        if prev_sl_pnl is None or float(sl_pnl) > float(prev_sl_pnl):
                            position.moved_sl_on_profit = True

                if move_tp_on_loss and move_tp_trigger_pnl is not None and move_tp_target_pnl is not None:
                    try:
                        trigger_pnl = float(move_tp_trigger_pnl)
                        target_pnl = float(move_tp_target_pnl)
                    except (TypeError, ValueError):
                        trigger_pnl = None
                        target_pnl = None
                    if (
                        trigger_pnl is not None
                        and target_pnl is not None
                        and target_pnl > 0
                        and trigger_pnl < 0
                        and position.min_return <= trigger_pnl
                    ):
                        prev_tp_pnl = tp_pnl
                        if tp_pnl is None:
                            tp_pnl = target_pnl
                        else:
                            tp_pnl = max(float(tp_pnl), target_pnl)
                        if prev_tp_pnl is None or float(tp_pnl) > float(prev_tp_pnl):
                            position.moved_tp_on_loss = True

                if entry_price > 0 and sl_pnl is not None:
                    sl_unlevered = float(sl_pnl) / leverage_value
                    if position.direction > 0:
                        sl_price = entry_price * (1.0 + sl_unlevered)
                        if low_price <= sl_price:
                            cash = self._close_position(
                                cash=cash,
                                position=position,
                                trades=trades,
                                exit_price=sl_price,
                                timestamp=timestamp,
                                exit_reason=(
                                    "stop_loss_moved"
                                    if position.moved_sl_on_profit
                                    else "stop_loss"
                                ),
                            )
                            return None
                    else:
                        sl_price = entry_price * (1.0 + abs(sl_unlevered))
                        if high_price >= sl_price:
                            cash = self._close_position(
                                cash=cash,
                                position=position,
                                trades=trades,
                                exit_price=sl_price,
                                timestamp=timestamp,
                                exit_reason=(
                                    "stop_loss_moved"
                                    if position.moved_sl_on_profit
                                    else "stop_loss"
                                ),
                            )
                            return None
                if entry_price > 0 and tp_pnl is not None:
                    tp_unlevered = float(tp_pnl) / leverage_value
                    if position.direction > 0:
                        tp_price = entry_price * (1.0 + tp_unlevered)
                        if high_price >= tp_price:
                            cash = self._close_position(
                                cash=cash,
                                position=position,
                                trades=trades,
                                exit_price=tp_price,
                                timestamp=timestamp,
                                exit_reason=(
                                    "take_profit_moved"
                                    if position.moved_tp_on_loss
                                    else "take_profit"
                                ),
                            )
                            return None
                    else:
                        tp_price = entry_price * (1.0 - tp_unlevered)
                        if low_price <= tp_price:
                            cash = self._close_position(
                                cash=cash,
                                position=position,
                                trades=trades,
                                exit_price=tp_price,
                                timestamp=timestamp,
                                exit_reason=(
                                    "take_profit_moved"
                                    if position.moved_tp_on_loss
                                    else "take_profit"
                                ),
                            )
                            return None

            if time_stop_bars is not None and time_stop_min_r is not None:
                bars_in_trade = i - position.entry_index
                if bars_in_trade >= int(time_stop_bars) and position.risk_pct > 0:
                    if position.max_return < position.risk_pct * float(time_stop_min_r):
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=close_price,
                            timestamp=timestamp,
                            exit_reason="time_stop",
                        )
                        return None

            if use_timeout_cross and timeout_cross_bars is not None and timeout_sell_pnl is not None:
                bars_since_cross = i - position.last_signal_index
                if bars_since_cross >= int(timeout_cross_bars):
                    target_price = _target_price_for_pnl(
                        entry_price,
                        position.direction,
                        leverage_value,
                        float(timeout_sell_pnl),
                    )
                    if target_price is not None and (
                        (position.direction > 0 and high_price >= target_price)
                        or (position.direction < 0 and low_price <= target_price)
                    ):
                        cash = self._close_position(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=close_price,
                            timestamp=timestamp,
                            exit_reason="timeout_exit",
                        )
                        return None

            if partial_exit_mode in ("all", "added") and position.added_qty > 0:
                partial_pnl = long_partial_exit_pnl if position.direction > 0 else short_partial_exit_pnl
                if partial_pnl is not None:
                    target_price = _target_price_for_pnl(
                        entry_price,
                        position.direction,
                        leverage_value,
                        float(partial_pnl),
                    )
                    if target_price is not None and (
                        (position.direction > 0 and high_price >= target_price)
                        or (position.direction < 0 and low_price <= target_price)
                    ):
                        if partial_exit_mode == "all":
                            cash = self._close_position(
                                cash=cash,
                                position=position,
                                trades=trades,
                                exit_price=close_price,
                                timestamp=timestamp,
                                exit_reason="partial_exit_all",
                            )
                            return None
                        cash = self._close_partial(
                            cash=cash,
                            position=position,
                            trades=trades,
                            exit_price=close_price,
                            qty_to_close=position.added_qty,
                            timestamp=timestamp,
                            exit_reason="partial_exit_added",
                        )

            exit_signal = int(exit_signals.iloc[i])
            if exit_signal != 0 and exit_signal == position.direction:
                can_exit = True
                target_price = None
                if slope_exit_mode:
                    if slope_exit_cooldown_bars is not None:
                        can_exit = position.slope_exit_cooldown == 0
                    if can_exit:
                        if not slope_exit_use_threshold:
                            threshold_ok = True
                        elif slope_exit_ignore_threshold_after_add and position.parts_filled > 1:
                            threshold_ok = True
                        else:
                            if slope_exit_pnl_threshold is None:
                                threshold_ok = True
                            else:
                                target_price = _target_price_for_pnl(
                                    entry_price,
                                    position.direction,
                                    leverage_value,
                                    float(slope_exit_pnl_threshold),
                                )
                                threshold_ok = target_price is not None and (
                                    (position.direction > 0 and high_price >= target_price)
                                    or (position.direction < 0 and low_price <= target_price)
                                )
                        can_exit = threshold_ok
                if can_exit:
                    cash = self._close_position(
                        cash=cash,
                        position=position,
                        trades=trades,
                        exit_price=close_price,
                        timestamp=timestamp,
                        exit_reason="slope_exit",
                    )
                    return None

            return position

        add_buy_debug: Dict[str, Any] = {
            "attempts": 0,
            "triggered": 0,
            "executed": 0,
            "blocked": {},
            "attempts_by_direction": {1: 0, -1: 0},
            "triggered_by_direction": {1: 0, -1: 0},
            "executed_by_direction": {1: 0, -1: 0},
            "blocked_by_direction": {1: {}, -1: {}},
        }

        def _dbg_incr(reason: str | None, direction: int, bucket: str = "blocked") -> None:
            if bucket == "attempts":
                add_buy_debug["attempts"] += 1
                add_buy_debug["attempts_by_direction"][direction] = (
                    add_buy_debug["attempts_by_direction"].get(direction, 0) + 1
                )
                return
            if bucket == "triggered":
                add_buy_debug["triggered"] += 1
                add_buy_debug["triggered_by_direction"][direction] = (
                    add_buy_debug["triggered_by_direction"].get(direction, 0) + 1
                )
                return
            if bucket == "executed":
                add_buy_debug["executed"] += 1
                add_buy_debug["executed_by_direction"][direction] = (
                    add_buy_debug["executed_by_direction"].get(direction, 0) + 1
                )
                return
            if reason is None:
                return
            add_buy_debug["blocked"][reason] = add_buy_debug["blocked"].get(reason, 0) + 1
            per_dir = add_buy_debug["blocked_by_direction"].get(direction)
            if per_dir is not None:
                per_dir[reason] = per_dir.get(reason, 0) + 1

        def _process_add_buy(
            position: Optional[Position],
            i: int,
            row: pd.Series,
            close_price: float,
            high_price: float,
            low_price: float,
            timestamp: pd.Timestamp,
            signal_now: int,
        ) -> Optional[Position]:
            nonlocal cash
            if position is None or position.parts_filled <= 0:
                return position
            if not use_add_buy:
                _dbg_incr("use_add_buy_disabled", position.direction)
                return position
            parts = position.base_parts if position.base_parts > 0 else _resolve_parts()
            if position.parts_filled >= parts:
                _dbg_incr("parts_completed", position.direction)
                return position
            max_additions = _resolve_max_additions()
            if max_additions is not None and (position.parts_filled - 1) >= max_additions:
                _dbg_incr("max_additions_reached", position.direction)
                return position
            if signal_now == -position.direction:
                _dbg_incr("opposite_signal", position.direction)
                return position

            _dbg_incr(None, position.direction, "attempts")
            add_triggered = False
            add_triggered_by_pnl = False
            add_buy_pnl = None
            if position.parts_filled >= 2 and use_second_buy and ema_cross_signals is not None:
                if int(ema_cross_signals.iloc[i]) == position.direction:
                    add_triggered = True
                else:
                    _dbg_incr("second_buy_signal_mismatch", position.direction)
            else:
                if use_prev_extreme_rr and add_buy_use_rr_ratio and add_buy_rr_ratio is not None:
                    try:
                        rr_ratio = float(add_buy_rr_ratio)
                    except (TypeError, ValueError):
                        rr_ratio = None
                    if rr_ratio is not None and rr_ratio > 0:
                        rr_stop_price = position.rr_stop_price
                        if rr_stop_price is None:
                            _dbg_incr("missing_rr_stop", position.direction)
                        else:
                            leverage_value = (
                                position.leverage if position.leverage > 0 else 1.0
                            )
                            rr_sl_pnl = (
                                self._directional_return(
                                    position.entry_price,
                                    rr_stop_price,
                                    position.direction,
                                )
                                * leverage_value
                            )
                            if rr_sl_pnl < 0:
                                add_buy_pnl = rr_sl_pnl * rr_ratio
                            else:
                                _dbg_incr("rr_stop_not_loss", position.direction)
                    else:
                        _dbg_incr("invalid_rr_ratio", position.direction)

                if add_buy_pnl is None:
                    add_buy_pnl = long_add_buy_pnl if position.direction > 0 else short_add_buy_pnl
                override_factor = 1.0
                if atr_override_factor is not None:
                    override_factor = float(atr_override_factor.iloc[i])
                if add_buy_pnl is not None:
                    add_buy_pnl = float(add_buy_pnl) * override_factor
                if add_buy_pnl is None:
                    _dbg_incr("missing_add_buy_pnl", position.direction)
                else:
                    leverage_value = position.leverage if position.leverage > 0 else 1.0
                    # Allow add-buy if current PnL is below threshold or intrabar touched it.
                    ret_pct_close = (
                        self._directional_return(position.entry_price, close_price, position.direction)
                        * leverage_value
                    )
                    trigger_price = low_price if position.direction > 0 else high_price
                    ret_pct_touch = (
                        self._directional_return(position.entry_price, trigger_price, position.direction)
                        * leverage_value
                    )
                    pnl_reached = ret_pct_close <= float(add_buy_pnl) or ret_pct_touch <= float(add_buy_pnl)
                    current_candle_ok = True
                    if add_buy_candle_condition:
                        if position.direction > 0:
                            current_candle_ok = close_price > float(row["open"])
                        else:
                            current_candle_ok = close_price < float(row["open"])
                    enforce_macro = False
                    enforce_candle = False
                    macro_ok = True
                    prev_candle_ok = True
                    if position.parts_filled == 1:
                        enforce_macro = use_first_add_macro_ema
                        enforce_candle = first_add_candle_condition_enabled
                        if bb_width_narrow is not None and bool(bb_width_narrow.iloc[i]):
                            enforce_macro = True
                            enforce_candle = True
                        if enforce_macro and macro_long_prev is not None and macro_short_prev is not None:
                            if position.direction > 0:
                                macro_ok = bool(macro_long_prev.iloc[i])
                            else:
                                macro_ok = bool(macro_short_prev.iloc[i])
                        if enforce_candle and "open" in self.data.columns:
                            prev_open = float(self.data.iloc[i - 1]["open"]) if i > 0 else float(row["open"])
                            prev_close = float(self.data.iloc[i - 1]["close"]) if i > 0 else close_price
                            if position.direction > 0:
                                prev_candle_ok = prev_close > prev_open
                            else:
                                prev_candle_ok = prev_close < prev_open

                    candle_ok = current_candle_ok and macro_ok and prev_candle_ok
                    if pnl_reached and candle_ok:
                        add_triggered = True
                        add_triggered_by_pnl = True
                    else:
                        if not pnl_reached:
                            _dbg_incr("pnl_not_reached", position.direction)
                        else:
                            if not current_candle_ok:
                                _dbg_incr("candle_condition_current", position.direction)
                            if enforce_macro and not macro_ok:
                                _dbg_incr("macro_ema_filter", position.direction)
                            if enforce_candle and not prev_candle_ok:
                                _dbg_incr("candle_condition_prev", position.direction)

            if add_triggered:
                _dbg_incr(None, position.direction, "triggered")
                # Add the current position size (doubling) on each add-buy.
                add_qty = position.qty
                add_price = close_price
                if add_triggered_by_pnl and add_buy_pnl is not None:
                    leverage_value = position.leverage if position.leverage > 0 else 1.0
                    target_price = _target_price_for_pnl(
                        position.entry_price,
                        position.direction,
                        leverage_value,
                        float(add_buy_pnl),
                    )
                    if target_price is not None:
                        add_price = target_price
                max_fraction = self.max_position_fraction_per_side
                if max_fraction is not None and max_fraction > 0:
                    max_qty = position.target_qty * float(max_fraction)
                    if position.qty + add_qty > max_qty:
                        _dbg_incr("max_position_fraction", position.direction)
                        return position
                if add_qty > 0:
                    add_notional = abs(add_qty * add_price)
                    add_fee = add_notional * self.fee_rate
                    leverage_value = position.leverage if position.leverage > 0 else 1.0
                    required_cash = (add_notional / leverage_value) + add_fee
                    if cash < required_cash:
                        _dbg_incr("insufficient_cash", position.direction)
                        return position
                    cash -= required_cash
                    position.added_qty += add_qty
                    position.added_cost += add_notional
                    position.parts_filled += 1
                    position.qty = position.base_qty + position.added_qty
                    if position.qty > 0:
                        position.entry_price = (position.base_cost + position.added_cost) / position.qty
                    _dbg_incr(None, position.direction, "executed")
                    trades.append(
                        {
                            "type": "add",
                            "direction": position.direction,
                            "price": add_price,
                            "qty": add_qty,
                            "fee": add_fee,
                            "timestamp": timestamp,
                            "reason": "add_buy",
                        }
                    )
                else:
                    _dbg_incr("zero_add_qty", position.direction)

            return position

        for i in range(len(self.data)):
            if progress_cb and (i % progress_step == 0 or i == total_rows - 1):
                progress_cb(i + 1, total_rows)
            row = self.data.iloc[i]
            timestamp = row["timestamp"] if "timestamp" in self.data.columns else i
            close_price = float(row["close"])
            high_price = float(row["high"]) if "high" in self.data.columns else close_price
            low_price = float(row["low"]) if "low" in self.data.columns else close_price

            equity = cash
            if long_position is not None:
                leverage_value = long_position.leverage if long_position.leverage > 0 else 1.0
                equity += (long_position.base_cost + long_position.added_cost) / leverage_value
                equity += _update_position_extremes(long_position, close_price, high_price, low_price)
            if short_position is not None:
                leverage_value = short_position.leverage if short_position.leverage > 0 else 1.0
                equity += (short_position.base_cost + short_position.added_cost) / leverage_value
                equity += _update_position_extremes(short_position, close_price, high_price, low_price)
            equity_rows.append({"timestamp": timestamp, "equity": equity})

            signal_now = int(signals.iloc[i])
            if long_position is not None:
                if signal_now == long_position.direction:
                    long_position.last_signal_index = i
                if long_position.slope_exit_cooldown > 0:
                    long_position.slope_exit_cooldown -= 1
            if short_position is not None:
                if signal_now == short_position.direction:
                    short_position.last_signal_index = i
                if short_position.slope_exit_cooldown > 0:
                    short_position.slope_exit_cooldown -= 1

            long_position = _process_exits(
                long_position, i, row, timestamp, close_price, high_price, low_price
            )
            short_position = _process_exits(
                short_position, i, row, timestamp, close_price, high_price, low_price
            )

            if i >= len(self.data) - 1:
                continue

            signal = int(signals.iloc[i])
            entered_long = False
            entered_short = False
            if signal == 1 and long_position is None:
                next_open = float(self.data.iloc[i + 1]["open"])
                next_open = next_open * (1.0 + self.slippage)
                entry_reason = reasons.iloc[i] if i < len(reasons) else ""
                entry_time = self.data.iloc[i + 1].get("timestamp", i + 1)
                long_position = _enter_position(
                    1, next_open, i + 1, entry_time, entry_reason, i
                )
                entered_long = long_position is not None
            elif signal == -1 and short_position is None:
                next_open = float(self.data.iloc[i + 1]["open"])
                next_open = next_open * (1.0 - self.slippage)
                entry_reason = reasons.iloc[i] if i < len(reasons) else ""
                entry_time = self.data.iloc[i + 1].get("timestamp", i + 1)
                short_position = _enter_position(
                    -1, next_open, i + 1, entry_time, entry_reason, i
                )
                entered_short = short_position is not None

            if long_position is not None and not entered_long:
                long_position = _process_add_buy(
                    long_position, i, row, close_price, high_price, low_price, timestamp, signal_now
                )
            if short_position is not None and not entered_short:
                short_position = _process_add_buy(
                    short_position, i, row, close_price, high_price, low_price, timestamp, signal_now
                )

        open_positions: Dict[str, Any] = {}
        if close_open_positions:
            if long_position is not None:
                last_row = self.data.iloc[-1]
                last_price = float(last_row["close"])
                cash = self._close_position(
                    cash=cash,
                    position=long_position,
                    trades=trades,
                    exit_price=last_price,
                    timestamp=last_row.get("timestamp", len(self.data) - 1),
                    exit_reason="end_of_data",
                )
            if short_position is not None:
                last_row = self.data.iloc[-1]
                last_price = float(last_row["close"])
                cash = self._close_position(
                    cash=cash,
                    position=short_position,
                    trades=trades,
                    exit_price=last_price,
                    timestamp=last_row.get("timestamp", len(self.data) - 1),
                    exit_reason="end_of_data",
                )
        else:
            if long_position is not None:
                open_positions["long"] = {
                    "direction": long_position.direction,
                    "entry_price": long_position.entry_price,
                    "qty": long_position.qty,
                    "leverage": long_position.leverage,
                    "stop_price": long_position.stop_price,
                    "prev_extreme_stop_price": long_position.prev_extreme_stop_price,
                    "rr_stop_price": long_position.rr_stop_price,
                    "rr_tp_price": long_position.rr_tp_price,
                    "entry_time": long_position.entry_time,
                    "entry_reason": long_position.entry_reason,
                    "min_return": long_position.min_return,
                    "max_return": long_position.max_return,
                }
            if short_position is not None:
                open_positions["short"] = {
                    "direction": short_position.direction,
                    "entry_price": short_position.entry_price,
                    "qty": short_position.qty,
                    "leverage": short_position.leverage,
                    "stop_price": short_position.stop_price,
                    "prev_extreme_stop_price": short_position.prev_extreme_stop_price,
                    "rr_stop_price": short_position.rr_stop_price,
                    "rr_tp_price": short_position.rr_tp_price,
                    "entry_time": short_position.entry_time,
                    "entry_reason": short_position.entry_reason,
                    "min_return": short_position.min_return,
                    "max_return": short_position.max_return,
                }

        equity_df = pd.DataFrame(equity_rows)
        return BacktestResult(
            equity_curve=equity_df,
            trades=trades,
            add_buy_debug=add_buy_debug,
            final_cash=cash,
            open_positions=open_positions,
        )

    def _calc_qty(
        self, equity: float, price: float, stop_distance: float | None, leverage_override: float | None = None
    ) -> float:
        if price <= 0:
            return 0.0

        leverage = leverage_override if leverage_override is not None else self.leverage
        position_size = self.position_size if self.position_size > 0 else 1.0
        notional = equity * position_size * leverage
        qty = (notional / price) * 0.99
        return qty

    def _directional_return(self, entry_price: float, current_price: float, direction: int) -> float:
        if entry_price <= 0:
            return 0.0
        return ((current_price - entry_price) / entry_price) * direction

    def _close_partial(
        self,
        cash: float,
        position: Position,
        trades: List[Dict[str, float]],
        exit_price: float,
        qty_to_close: float,
        timestamp: pd.Timestamp,
        exit_reason: str,
    ) -> float:
        if qty_to_close <= 0 or position.qty <= 0:
            return cash
        qty_to_close = min(qty_to_close, position.qty)
        entry_price = position.entry_price
        if position.added_qty > 0 and qty_to_close <= position.added_qty:
            entry_price = position.added_cost / position.added_qty if position.added_qty > 0 else entry_price
        exit_fee = abs(qty_to_close * exit_price) * self.fee_rate
        pnl = (exit_price - entry_price) * qty_to_close * position.direction
        leverage_value = position.leverage if position.leverage > 0 else 1.0
        closed_notional = abs(entry_price * qty_to_close)
        released_margin = closed_notional / leverage_value if leverage_value > 0 else closed_notional
        cash += pnl - exit_fee + released_margin
        trades.append(
            {
                "type": "exit_partial",
                "direction": position.direction,
                "entry_price": entry_price,
                "price": exit_price,
                "qty": qty_to_close,
                "fee": exit_fee,
                "leverage": position.leverage,
                "pnl": pnl,
                "return": self._directional_return(entry_price, exit_price, position.direction)
                * leverage_value,
                "timestamp": timestamp,
                "exit_reason": exit_reason,
                "stop_price": position.stop_price,
                "rr_stop_price": position.rr_stop_price,
                "rr_tp_price": position.rr_tp_price,
            }
        )
        if position.added_qty > 0:
            removed = min(position.added_qty, qty_to_close)
            position.added_qty -= removed
            if position.added_qty <= 0:
                position.added_cost = 0.0
            else:
                position.added_cost *= position.added_qty / (position.added_qty + removed)
        position.qty = position.base_qty + position.added_qty
        if position.qty > 0:
            position.entry_price = (position.base_cost + position.added_cost) / position.qty
        return cash

    def _close_position(
        self,
        cash: float,
        position: Position,
        trades: List[Dict[str, float]],
        exit_price: float,
        timestamp: pd.Timestamp,
        exit_reason: str,
    ) -> float:
        exit_fee = abs(position.qty * exit_price) * self.fee_rate
        pnl = (exit_price - position.entry_price) * position.qty * position.direction
        notional = abs(position.entry_price * position.qty)
        ret_pct = pnl / notional if notional > 0 else 0.0
        leverage_value = position.leverage if position.leverage > 0 else 1.0
        released_margin = notional / leverage_value if leverage_value > 0 else notional
        cash += pnl - exit_fee + released_margin
        trades.append(
            {
                "type": "exit",
                "direction": position.direction,
                "entry_price": position.entry_price,
                "price": exit_price,
                "qty": position.qty,
                "fee": exit_fee,
                "leverage": position.leverage,
                "pnl": pnl,
                "return": ret_pct * leverage_value,
                "min_return": position.min_return,
                "max_return": position.max_return,
                "entry_time": position.entry_time,
                "entry_reason": position.entry_reason,
                "exit_reason": exit_reason,
                "timestamp": timestamp,
                "stop_price": position.stop_price,
                "rr_stop_price": position.rr_stop_price,
                "rr_tp_price": position.rr_tp_price,
            }
        )
        return cash
