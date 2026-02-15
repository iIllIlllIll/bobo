from __future__ import annotations

import pandas as pd

from backtest.strategy import Strategy

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def _normalize_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0 if abs(value) >= 1.0 else value


def _normalize_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val / 100.0 if abs(val) >= 1.0 else val


def _normalize_buffer_pct(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        val = abs(float(value))
    except (TypeError, ValueError):
        return None
    return val / 100.0


def _normalize_pct_list(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [_normalize_pct(v) if v is not None else v for v in values]


class CheatkeyStrategy(Strategy):
    NAME = "cheatkey"

    def __init__(
        self,
        ema_fast: int = 12,
        ema_slow: int = 26,
        fast_window: int | None = None,
        slow_window: int | None = None,
        cheatkey_threshold: float = 0.001,
        cheatkey_threshold_use_pct: bool = False,
        cheatkey_threshold_pct: float = 0.001,
        cheatkey_lookback: int = 6,
        cheatkey_timefilter: bool = True,
        cheatkey_diff_lookback: int | None = 1,
        slope_exit_mode: bool = True,
        slope_exit: bool | None = None,
        slope_exit_lookback: int = 1,
        slope_exit_cooldown_bars: int = 20,
        slope_exit_pnl_threshold: float = 5.0,
        slope_exit_use_threshold: bool = True,
        slope_exit_ignore_threshold_after_add: bool = False,
        long_leverage: float = 10.0,
        short_leverage: float = 10.0,
        long_tp_pnl: float = 15.0,
        short_tp_pnl: float = 15.0,
        long_add_tp_pnl: list[float] | None = None,
        short_add_tp_pnl: list[float] | None = None,
        long_sl_pnl: float = -10.0,
        short_sl_pnl: float = -10.0,
        long_add_sl_pnl: list[float] | None = None,
        short_add_sl_pnl: list[float] | None = None,
        move_sl_on_profit: bool = False,
        move_sl_trigger_pnl: float | None = None,
        move_sl_target_pnl: float | None = None,
        move_sl_use_ratio: bool = False,
        move_sl_trigger_ratio: float | None = None,
        move_sl_target_ratio: float | None = None,
        move_tp_on_loss: bool = False,
        move_tp_trigger_pnl: float | None = None,
        move_tp_target_pnl: float | None = None,
        use_rr_tp_from_sl_pnl: bool = False,
        rr_tp_from_sl_multiplier: float | None = None,
        long_partial_exit_pnl: float = 10.0,
        short_partial_exit_pnl: float = 10.0,
        partial_exit_mode: str | None = "added",
        long_add_buy_pnl: float = -7.0,
        short_add_buy_pnl: float = -7.0,
        add_buy_candle_condition: bool = False,
        use_add_buy: bool = True,
        add_buy_max_times: int | None = None,
        use_second_buy: bool = True,
        divide_value: int = 3,
        total_parts: int | None = None,
        exit_mode: str | None = "all",
        exit_full_condition: str = "always",
        first_opposite_exit: bool = True,
        first_opposite_exit_pnl: float = 0.0,
        exitmode_timefilter: bool = True,
        use_timeout_cross: bool = False,
        timeout_cross_bars: int = 5,
        timeout_sell_pnl: float = 0.01,
        use_prev_extreme_stop: bool = False,
        prev_extreme_lookback: int = 20,
        prev_extreme_buffer_pct: float = 0.1,
        prev_extreme_max_loss_pct: float | None = None,
        use_prev_extreme_rr: bool = False,
        prev_extreme_rr_lookback: int = 20,
        prev_extreme_rr_buffer_pct: float = 0.1,
        prev_extreme_rr_multiplier: float = 2.0,
        prev_extreme_rr_max_loss_pct: float | None = None,
        use_prev_extreme_rr_tp_cap: bool = False,
        prev_extreme_rr_tp_cap_lookback: int = 20,
        prev_extreme_rr_tp_cap_buffer_pct: float = 0.1,
        macro_ema_filter: bool = True,
        macro_ema_window: int = 100,
        macro_ema_period: int | None = None,
        macro_ema_align_filter: bool = False,
        macro_ema_align_windows: list[int] | None = None,
        ema_dir_filter: bool = True,
        ema_dir_window: int = 50,
        ema_dir_lookback: int = 6,
        use_4h_candle_condition: bool = True,
        use_first_add_macro_ema: bool = True,
        first_add_candle_condition_enabled: bool = True,
        use_override_atr: bool = True,
        atr_override_period: int = 16,
        atr_pct_threshold: float = 0.6,
        unit_override: float = 0.8,
        bb_width_window: int = 20,
        bb_width_std: float = 2.0,
        bb_width_threshold: float = 0.05,
        **_kwargs: object,
    ) -> None:
        if fast_window is not None:
            ema_fast = fast_window
        if slow_window is not None:
            ema_slow = slow_window
        if macro_ema_period is not None:
            macro_ema_window = macro_ema_period
        if macro_ema_align_windows is None:
            macro_ema_align_windows = [50, 100, 200, 400]
        if macro_ema_align_filter:
            for window in macro_ema_align_windows:
                if int(window) <= 0:
                    raise ValueError("macro_ema_align_windows must be positive integers")
        if ema_fast >= ema_slow:
            raise ValueError("ema_fast must be less than ema_slow")
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cheatkey_threshold = cheatkey_threshold
        self.cheatkey_threshold_use_pct = cheatkey_threshold_use_pct
        self.cheatkey_threshold_pct = _normalize_buffer_pct(cheatkey_threshold_pct) or 0.0
        self.cheatkey_lookback = cheatkey_lookback
        self.cheatkey_timefilter = cheatkey_timefilter
        self.cheatkey_diff_lookback = 1 if cheatkey_diff_lookback is None else cheatkey_diff_lookback
        self.slope_exit_mode = slope_exit_mode if slope_exit is None else slope_exit
        self.slope_exit_lookback = slope_exit_lookback
        self.slope_exit_cooldown_bars = slope_exit_cooldown_bars
        self.slope_exit_pnl_threshold = _normalize_pct(slope_exit_pnl_threshold)
        self.slope_exit_use_threshold = slope_exit_use_threshold
        self.slope_exit_ignore_threshold_after_add = slope_exit_ignore_threshold_after_add
        self.long_leverage = long_leverage
        self.short_leverage = short_leverage
        self.long_tp_pnl = _normalize_pct(long_tp_pnl)
        self.short_tp_pnl = _normalize_pct(short_tp_pnl)
        self.long_add_tp_pnl = _normalize_pct_list(long_add_tp_pnl)
        self.short_add_tp_pnl = _normalize_pct_list(short_add_tp_pnl)
        self.long_sl_pnl = _normalize_pct(long_sl_pnl)
        self.short_sl_pnl = _normalize_pct(short_sl_pnl)
        self.long_add_sl_pnl = _normalize_pct_list(long_add_sl_pnl) or [-15.0, -15.0, -15.0]
        self.short_add_sl_pnl = _normalize_pct_list(short_add_sl_pnl) or [-15.0, -15.0, -15.0]
        self.move_sl_on_profit = move_sl_on_profit
        self.move_sl_trigger_pnl = _normalize_pct(move_sl_trigger_pnl)
        self.move_sl_target_pnl = _normalize_pct(move_sl_target_pnl)
        self.move_sl_use_ratio = move_sl_use_ratio
        self.move_sl_trigger_ratio = _normalize_ratio(move_sl_trigger_ratio)
        self.move_sl_target_ratio = _normalize_ratio(move_sl_target_ratio)
        self.move_tp_on_loss = move_tp_on_loss
        self.move_tp_trigger_pnl = _normalize_pct(move_tp_trigger_pnl)
        self.move_tp_target_pnl = _normalize_pct(move_tp_target_pnl)
        self.use_rr_tp_from_sl_pnl = use_rr_tp_from_sl_pnl
        try:
            self.rr_tp_from_sl_multiplier = float(rr_tp_from_sl_multiplier)
        except (TypeError, ValueError):
            self.rr_tp_from_sl_multiplier = None
        self.long_partial_exit_pnl = _normalize_pct(long_partial_exit_pnl)
        self.short_partial_exit_pnl = _normalize_pct(short_partial_exit_pnl)
        self.partial_exit_mode = partial_exit_mode
        self.long_add_buy_pnl = _normalize_pct(long_add_buy_pnl)
        self.short_add_buy_pnl = _normalize_pct(short_add_buy_pnl)
        self.add_buy_candle_condition = add_buy_candle_condition
        self.use_add_buy = use_add_buy
        self.add_buy_max_times = add_buy_max_times
        self.use_second_buy = use_second_buy
        self.divide_value = divide_value
        self.total_parts = total_parts
        self.exit_mode = exit_mode
        self.exit_full_condition = exit_full_condition
        self.first_opposite_exit = first_opposite_exit
        self.first_opposite_exit_pnl = _normalize_pct(first_opposite_exit_pnl)
        self.exitmode_timefilter = exitmode_timefilter
        self.use_timeout_cross = use_timeout_cross
        self.timeout_cross_bars = timeout_cross_bars
        self.timeout_sell_pnl = _normalize_pct(timeout_sell_pnl)
        self.use_prev_extreme_stop = use_prev_extreme_stop
        self.prev_extreme_lookback = int(prev_extreme_lookback)
        self.prev_extreme_buffer_pct = _normalize_buffer_pct(prev_extreme_buffer_pct) or 0.0
        self.prev_extreme_max_loss_pct = _normalize_pct(prev_extreme_max_loss_pct)
        self.use_prev_extreme_rr = use_prev_extreme_rr
        self.prev_extreme_rr_lookback = int(prev_extreme_rr_lookback)
        self.prev_extreme_rr_buffer_pct = _normalize_buffer_pct(prev_extreme_rr_buffer_pct) or 0.0
        try:
            self.prev_extreme_rr_multiplier = float(prev_extreme_rr_multiplier)
        except (TypeError, ValueError):
            self.prev_extreme_rr_multiplier = 0.0
        self.prev_extreme_rr_max_loss_pct = _normalize_pct(prev_extreme_rr_max_loss_pct)
        self.use_prev_extreme_rr_tp_cap = use_prev_extreme_rr_tp_cap
        self.prev_extreme_rr_tp_cap_lookback = int(prev_extreme_rr_tp_cap_lookback)
        self.prev_extreme_rr_tp_cap_buffer_pct = (
            _normalize_buffer_pct(prev_extreme_rr_tp_cap_buffer_pct) or 0.0
        )
        self.macro_ema_filter = macro_ema_filter
        self.macro_ema_window = macro_ema_window
        self.macro_ema_align_filter = macro_ema_align_filter
        self.macro_ema_align_windows = list(macro_ema_align_windows)
        self.ema_dir_filter = ema_dir_filter
        self.ema_dir_window = int(ema_dir_window)
        self.ema_dir_lookback = int(ema_dir_lookback)
        self.use_4h_candle_condition = use_4h_candle_condition
        self.use_first_add_macro_ema = use_first_add_macro_ema
        self.first_add_candle_condition_enabled = first_add_candle_condition_enabled
        self.use_override_atr = use_override_atr
        self.atr_override_period = atr_override_period
        self.atr_pct_threshold = atr_pct_threshold
        self.unit_override = unit_override
        self.bb_width_window = bb_width_window
        self.bb_width_std = bb_width_std
        self.bb_width_threshold = bb_width_threshold
        if self.ema_dir_window <= 0:
            raise ValueError("ema_dir_window must be positive")
        if self.ema_dir_lookback <= 0:
            raise ValueError("ema_dir_lookback must be positive")

    def _compute(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain a 'timestamp' column")
        required = {"open", "high", "low", "close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Data must contain columns: {sorted(missing)}")

        base = data.sort_values("timestamp").reset_index(drop=True)
        base_index = base.index
        close = base["close"]
        timestamps = pd.to_datetime(base["timestamp"])

        ema_fast = _ema(close, self.ema_fast)
        ema_slow = _ema(close, self.ema_slow)
        ema_dir = _ema(close, self.ema_dir_window) if self.ema_dir_filter else None

        prev_fast = ema_fast.shift(1)
        prev_slow = ema_slow.shift(1)
        cross_up = (ema_fast > ema_slow) & (prev_fast <= prev_slow)
        cross_down = (ema_fast < ema_slow) & (prev_fast >= prev_slow)

        macro_long = pd.Series(True, index=base_index)
        macro_short = pd.Series(True, index=base_index)
        if self.macro_ema_filter and self.macro_ema_window > 0:
            macro_ema = _ema(close, self.macro_ema_window)
            macro_long = close > macro_ema
            macro_short = close < macro_ema

        if self.macro_ema_align_filter and self.macro_ema_align_windows:
            ema_series = [_ema(close, int(window)) for window in self.macro_ema_align_windows]
            macro_align_long = pd.Series(True, index=base_index)
            macro_align_short = pd.Series(True, index=base_index)
            for idx in range(len(ema_series) - 1):
                macro_align_long &= ema_series[idx] > ema_series[idx + 1]
                macro_align_short &= ema_series[idx] < ema_series[idx + 1]
            macro_long &= macro_align_long
            macro_short &= macro_align_short

        open_values = base["open"].to_numpy()
        close_values = close.to_numpy()
        ema_fast_values = ema_fast.to_numpy()
        ema_slow_values = ema_slow.to_numpy()
        ema_dir_values = ema_dir.to_numpy() if ema_dir is not None else None
        macro_long_values = macro_long.to_numpy()
        macro_short_values = macro_short.to_numpy()
        timestamp_values = pd.to_datetime(timestamps).to_numpy()

        lookback_n = max(int(self.cheatkey_lookback), 1)
        diff_lookback_n = max(int(self.cheatkey_diff_lookback or 1), 1)
        ema_dir_min = 0
        if self.ema_dir_filter:
            ema_dir_min = (self.ema_dir_window - 1) + self.ema_dir_lookback
        min_index = max(lookback_n, diff_lookback_n, 4, ema_dir_min)

        def _ema_dir_ok(idx: int, side: str) -> bool:
            if not self.ema_dir_filter:
                return True
            if idx - self.ema_dir_lookback < 0 or ema_dir_values is None:
                return False
            for j in range(self.ema_dir_lookback):
                curr = ema_dir_values[idx - j]
                prev = ema_dir_values[idx - j - 1]
                if pd.isna(curr) or pd.isna(prev):
                    return False
                if side == "long" and curr <= prev:
                    return False
                if side == "short" and curr >= prev:
                    return False
            return True

        def _cheatkey_signal(idx: int, side: str) -> bool:
            if self.cheatkey_timefilter:
                minute = pd.Timestamp(timestamp_values[idx]).minute
                if minute not in (0, 15, 30, 45):
                    return False
            if idx < min_index:
                return False
            if not _ema_dir_ok(idx, side):
                return False

            c12 = ema_fast_values[idx]
            c26 = ema_slow_values[idx]
            p12 = ema_fast_values[idx - 1]
            p26 = ema_slow_values[idx - 1]
            if pd.isna(c12) or pd.isna(c26) or pd.isna(p12) or pd.isna(p26):
                return False

            slope12 = c12 - p12
            slope26 = c26 - p26
            if side == "long" and (slope12 <= 0 or slope26 <= 0):
                return False
            if side == "short" and (slope12 >= 0 or slope26 >= 0):
                return False

            start = idx - lookback_n + 1
            for j in range(start, idx + 1):
                prev12 = ema_fast_values[j - 1]
                prev26 = ema_slow_values[j - 1]
                curr12 = ema_fast_values[j]
                curr26 = ema_slow_values[j]
                if pd.isna(prev12) or pd.isna(prev26) or pd.isna(curr12) or pd.isna(curr26):
                    return False
                if side == "long" and prev12 > prev26 and curr12 < curr26:
                    return False
                if side == "short" and prev12 < prev26 and curr12 > curr26:
                    return False

            if side == "long":
                if not (c12 < c26 and close_values[idx] > open_values[idx]):
                    return False
                for j in range(start, idx + 1):
                    if pd.isna(ema_fast_values[j]) or pd.isna(ema_slow_values[j]):
                        return False
                    if ema_fast_values[j] >= ema_slow_values[j]:
                        return False
            else:
                if not (c12 > c26 and close_values[idx] < open_values[idx]):
                    return False
                for j in range(start, idx + 1):
                    if pd.isna(ema_fast_values[j]) or pd.isna(ema_slow_values[j]):
                        return False
                    if ema_fast_values[j] <= ema_slow_values[j]:
                        return False

            diff_i = c12 - c26
            for k in range(1, diff_lookback_n + 1):
                j = idx - k
                prev_diff = ema_fast_values[j] - ema_slow_values[j]
                if pd.isna(prev_diff):
                    return False
                if diff_i * prev_diff <= 0 or abs(diff_i) >= abs(prev_diff):
                    return False
                diff_i = prev_diff

            threshold = float(self.cheatkey_threshold)
            if self.cheatkey_threshold_use_pct:
                threshold = abs(float(close_values[idx])) * float(self.cheatkey_threshold_pct)
            return abs(c12 - c26) <= threshold

        raw_long = pd.Series(False, index=base_index, dtype=bool)
        raw_short = pd.Series(False, index=base_index, dtype=bool)
        for idx in range(len(base_index)):
            if _cheatkey_signal(idx, "long") and macro_long_values[idx]:
                raw_long.iat[idx] = True
            if _cheatkey_signal(idx, "short") and macro_short_values[idx]:
                raw_short.iat[idx] = True

        signals = pd.Series(0, index=base_index, dtype=int)
        signals[raw_long] = 1
        signals[raw_short] = -1

        reasons = pd.Series("", index=base_index, dtype=object)
        reasons[raw_long] = "cheatkey_long_expected"
        reasons[raw_short] = "cheatkey_short_expected"

        return signals.reindex(data.index).fillna(0), reasons.reindex(data.index).fillna("")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals, _ = self._compute(data)
        return signals

    def signal_reasons(self, data: pd.DataFrame) -> pd.Series:
        _, reasons = self._compute(data)
        return reasons

    def ema_cross_signals(self, data: pd.DataFrame) -> pd.Series:
        if "timestamp" not in data.columns or "close" not in data.columns:
            raise ValueError("Data must contain 'timestamp' and 'close' columns")
        base = data.sort_values("timestamp").reset_index(drop=True)
        ema_fast = _ema(base["close"], self.ema_fast)
        ema_slow = _ema(base["close"], self.ema_slow)
        prev_fast = ema_fast.shift(1)
        prev_slow = ema_slow.shift(1)
        cross_up = (ema_fast > ema_slow) & (prev_fast <= prev_slow)
        cross_down = (ema_fast < ema_slow) & (prev_fast >= prev_slow)
        signals = pd.Series(0, index=base.index, dtype=int)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals.reindex(data.index).fillna(0)

    def exit_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.slope_exit_mode:
            return pd.Series([0] * len(data), index=data.index)
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain a 'timestamp' column")
        if "close" not in data.columns:
            raise ValueError("Data must contain a 'close' column")

        base = data.sort_values("timestamp").reset_index(drop=True)
        ema_fast = _ema(base["close"], self.ema_fast)
        ema_slow = _ema(base["close"], self.ema_slow)
        ema_fast_values = ema_fast.to_numpy()
        ema_slow_values = ema_slow.to_numpy()
        lookback = max(int(self.slope_exit_lookback), 1)

        def _ema_slope_ok(idx: int, side: str) -> bool:
            sign = -1 if side == "long" else 1
            for j in range(lookback):
                curr12 = ema_fast_values[idx - j]
                prev12 = ema_fast_values[idx - j - 1]
                curr26 = ema_slow_values[idx - j]
                prev26 = ema_slow_values[idx - j - 1]
                if pd.isna(curr12) or pd.isna(prev12) or pd.isna(curr26) or pd.isna(prev26):
                    return False
                if sign * (curr12 - prev12) <= 0:
                    return False
                if sign * (curr26 - prev26) <= 0:
                    return False
            return True

        exit_series = pd.Series(0, index=base.index, dtype=int)
        for idx in range(lookback, len(base)):
            if _ema_slope_ok(idx, "long"):
                exit_series.iat[idx] = 1
            elif _ema_slope_ok(idx, "short"):
                exit_series.iat[idx] = -1

        return exit_series.reindex(data.index).fillna(0)
