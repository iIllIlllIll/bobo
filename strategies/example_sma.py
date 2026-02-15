from __future__ import annotations

import pandas as pd

from backtest.strategy import Strategy
from backtest.timeframe import normalize_pandas_rule


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


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


def _adx(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(window, min_periods=window).sum()
    plus_dm_sum = plus_dm.rolling(window, min_periods=window).sum()
    minus_dm_sum = minus_dm.rolling(window, min_periods=window).sum()

    plus_di = 100 * (plus_dm_sum / tr_sum)
    minus_di = 100 * (minus_dm_sum / tr_sum)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100
    return dx.rolling(window, min_periods=window).mean()


def _resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for resampling: {sorted(missing)}")
    if "timestamp" not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column for resampling")

    df = data.set_index("timestamp").sort_index()
    rule = normalize_pandas_rule(rule)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    resampled = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return resampled


def _align(series: pd.Series, base_timestamps: pd.Index, base_index: pd.Index) -> pd.Series:
    aligned = series.reindex(base_timestamps, method="ffill")
    aligned.index = base_index
    return aligned


class SmaCrossStrategy(Strategy):
    NAME = "sma_cross"

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 30,
        adx_window: int = 14,
        adx_trend: int = 20,
        adx_tf: str = "15min",
        atr_window: int = 14,
        cross_atr_k: float = 0.2,
        cross_cooldown_bars: int = 2,
        ema_trend_window: int = 200,
        ema_trend_tf: str = "1h",
        ema_trend_slope_min: float = 0.0005,
        atr_slope_window: int = 3,
        atr_slope_min: float = 0.001,
        stop_atr_mult: float = 0.3,
        swing_lookback: int = 5,
        time_stop_bars: int = 8,
        time_stop_min_r: float = 0.5,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.adx_window = adx_window
        self.adx_trend = adx_trend
        self.adx_tf = adx_tf
        self.atr_window = atr_window
        self.cross_atr_k = cross_atr_k
        self.cross_cooldown_bars = cross_cooldown_bars
        self.ema_trend_window = ema_trend_window
        self.ema_trend_tf = ema_trend_tf
        self.ema_trend_slope_min = ema_trend_slope_min
        self.atr_slope_window = atr_slope_window
        self.atr_slope_min = atr_slope_min
        self.stop_atr_mult = stop_atr_mult
        self.swing_lookback = swing_lookback
        self.time_stop_bars = time_stop_bars
        self.time_stop_min_r = time_stop_min_r

    def _compute(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain a 'timestamp' column")
        required = {"open", "high", "low", "close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Data must contain columns: {sorted(missing)}")

        base = data.sort_values("timestamp").reset_index(drop=True)
        base_index = base.index
        base_timestamps = pd.Index(base["timestamp"])
        close = base["close"]

        fast = close.rolling(self.fast_window, min_periods=self.fast_window).mean()
        slow = close.rolling(self.slow_window, min_periods=self.slow_window).mean()
        raw_signal = (fast > slow).astype(int) - (fast < slow).astype(int)

        atr = _atr(base, self.atr_window)
        distance_ok = (fast - slow).abs() > self.cross_atr_k * atr

        adx_df = _resample_ohlcv(data, self.adx_tf)
        adx = _adx(adx_df, self.adx_window)
        adx_aligned = _align(adx, base_timestamps, base_index)

        trend_df = _resample_ohlcv(data, self.ema_trend_tf)
        ema_trend = _ema(trend_df["close"], self.ema_trend_window)
        ema_trend_aligned = _align(ema_trend, base_timestamps, base_index)
        ema_slope = ema_trend_aligned.pct_change()
        slope_ok = ema_slope.abs() >= self.ema_trend_slope_min

        atr_slope_ok = pd.Series(True, index=base.index)
        if self.atr_slope_min is not None and self.atr_slope_window > 0:
            atr_slope = atr.pct_change(self.atr_slope_window)
            atr_slope_ok = atr_slope >= self.atr_slope_min

        trend_ok = (adx_aligned >= self.adx_trend) & (slope_ok | atr_slope_ok)
        long_ok = trend_ok & (close > ema_trend_aligned)
        short_ok = trend_ok & (close < ema_trend_aligned)

        raw_long = (raw_signal == 1) & distance_ok & long_ok
        raw_short = (raw_signal == -1) & distance_ok & short_ok

        if self.cross_cooldown_bars > 0:
            recent_short = (
                raw_short.shift(1)
                .rolling(self.cross_cooldown_bars, min_periods=1)
                .max()
                .astype(bool)
            )
            recent_long = (
                raw_long.shift(1)
                .rolling(self.cross_cooldown_bars, min_periods=1)
                .max()
                .astype(bool)
            )
            raw_long = raw_long & ~recent_short
            raw_short = raw_short & ~recent_long

        signals = pd.Series(0, index=base.index, dtype=int)
        signals[raw_long] = 1
        signals[raw_short] = -1

        reasons = pd.Series("", index=base.index, dtype=object)
        reasons[raw_long] = "sma_cross_long_trend"
        reasons[raw_short] = "sma_cross_short_trend"

        return signals.reindex(data.index).fillna(0), reasons.reindex(data.index).fillna("")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals, _ = self._compute(data)
        return signals

    def signal_reasons(self, data: pd.DataFrame) -> pd.Series:
        _, reasons = self._compute(data)
        return reasons
