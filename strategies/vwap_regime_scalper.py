from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.strategy import Strategy
from backtest.timeframe import normalize_pandas_rule


@dataclass
class _ResampleSpec:
    rule: str
    ema_fast: int
    ema_slow: int
    ema_trend: int | None = None


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


def _vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return df["close"]
    volume = df["volume"].fillna(0.0)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical_price * volume).cumsum()
    vol = volume.cumsum().replace(0, pd.NA)
    return (pv / vol).fillna(df["close"])


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


def _align(series: pd.Series, index: pd.Index) -> pd.Series:
    return series.reindex(index, method="ffill")


class VwapRegimeScalperStrategy(Strategy):
    NAME = "vwap_regime_scalper"

    def __init__(
        self,
        adx_trend: int = 20,
        adx_range: int = 18,
        adx_window: int = 14,
        atr_window: int = 14,
        ema_fast: int = 20,
        ema_slow: int = 50,
        ema_trend: int = 200,
        ema_flat_threshold: float = 0.0015,
        vwap_band_atr: float = 1.5,
        base_ema: int = 20,
        require_reversal: bool = True,
        base_tf: str = "1min",
        trend_tf: str = "15min",
        confirm_tf: str = "1h",
    ) -> None:
        self.adx_trend = adx_trend
        self.adx_range = adx_range
        self.adx_window = adx_window
        self.atr_window = atr_window
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.ema_flat_threshold = ema_flat_threshold
        self.vwap_band_atr = vwap_band_atr
        self.base_ema = base_ema
        self.require_reversal = require_reversal
        self.base_tf = base_tf
        self.trend_tf = trend_tf
        self.confirm_tf = confirm_tf

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain a 'timestamp' column")
        required = {"open", "high", "low", "close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Data must contain columns: {sorted(missing)}")

        base = data.set_index("timestamp").sort_index()
        base_close = base["close"]
        base_open = base["open"]
        base_high = base["high"]
        base_low = base["low"]
        base_ema = _ema(base_close, self.base_ema)
        base_vwap = _vwap(base)
        base_atr = _atr(base, self.atr_window)

        trend_spec = _ResampleSpec(self.trend_tf, self.ema_fast, self.ema_slow)
        confirm_spec = _ResampleSpec(self.confirm_tf, self.ema_fast, self.ema_slow, self.ema_trend)

        trend_df = _resample_ohlcv(data, trend_spec.rule)
        confirm_df = _resample_ohlcv(data, confirm_spec.rule)

        trend_close = trend_df["close"]
        trend_ema_fast = _ema(trend_close, trend_spec.ema_fast)
        trend_ema_slow = _ema(trend_close, trend_spec.ema_slow)
        trend_vwap = _vwap(trend_df)
        trend_adx = _adx(trend_df, self.adx_window)

        confirm_close = confirm_df["close"]
        confirm_ema_trend = _ema(confirm_close, confirm_spec.ema_trend or self.ema_trend)
        confirm_adx = _adx(confirm_df, self.adx_window)

        base_index = base.index
        trend_ema_fast_aligned = _align(trend_ema_fast, base_index)
        trend_ema_slow_aligned = _align(trend_ema_slow, base_index)
        trend_vwap_aligned = _align(trend_vwap, base_index)
        trend_adx_aligned = _align(trend_adx, base_index)

        confirm_close_aligned = _align(confirm_close, base_index)
        confirm_ema_trend_aligned = _align(confirm_ema_trend, base_index)
        confirm_adx_aligned = _align(confirm_adx, base_index)

        ema_spread = (trend_ema_fast_aligned - trend_ema_slow_aligned).abs() / base_close

        trend_regime = (
            (trend_adx_aligned >= self.adx_trend)
            & (confirm_adx_aligned >= self.adx_trend)
            & (ema_spread >= self.ema_flat_threshold)
        )
        range_regime = (
            (trend_adx_aligned < self.adx_range)
            & (confirm_adx_aligned < self.adx_range)
            & (ema_spread < self.ema_flat_threshold)
        )

        trend_long = (
            trend_regime
            & (confirm_close_aligned > confirm_ema_trend_aligned)
            & (trend_ema_fast_aligned > trend_ema_slow_aligned)
            & (base_close > trend_vwap_aligned)
        )
        trend_short = (
            trend_regime
            & (confirm_close_aligned < confirm_ema_trend_aligned)
            & (trend_ema_fast_aligned < trend_ema_slow_aligned)
            & (base_close < trend_vwap_aligned)
        )

        prev_close = base_close.shift(1)
        prev_high = base_high.shift(1)
        prev_low = base_low.shift(1)
        pullback_long = (
            (prev_close <= base_vwap) & (base_close > base_vwap)
        ) | ((prev_close <= base_ema) & (base_close > base_ema))
        pullback_short = (
            (prev_close >= base_vwap) & (base_close < base_vwap)
        ) | ((prev_close >= base_ema) & (base_close < base_ema))

        if self.require_reversal:
            bullish = base_close > base_open
            bearish = base_close < base_open
            fail_new_low = base_low >= prev_low
            fail_new_high = base_high <= prev_high
            pullback_long = pullback_long & bullish & fail_new_low
            pullback_short = pullback_short & bearish & fail_new_high

        trend_long_signal = trend_long & pullback_long
        trend_short_signal = trend_short & pullback_short

        band = self.vwap_band_atr * base_atr
        range_long_signal = range_regime & (base_close <= base_vwap - band)
        range_short_signal = range_regime & (base_close >= base_vwap + band)

        signals = pd.Series(0, index=base_index, dtype=int)
        signals[range_long_signal] = 1
        signals[range_short_signal] = -1
        signals[trend_long_signal] = 1
        signals[trend_short_signal] = -1

        return signals.reindex(data.index, method="ffill").fillna(0)

    def signal_reasons(self, data: pd.DataFrame) -> pd.Series:
        if "timestamp" not in data.columns:
            raise ValueError("Data must contain a 'timestamp' column")
        required = {"open", "high", "low", "close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Data must contain columns: {sorted(missing)}")

        base = data.set_index("timestamp").sort_index()
        base_close = base["close"]
        base_open = base["open"]
        base_high = base["high"]
        base_low = base["low"]
        base_ema = _ema(base_close, self.base_ema)
        base_vwap = _vwap(base)
        base_atr = _atr(base, self.atr_window)

        trend_spec = _ResampleSpec(self.trend_tf, self.ema_fast, self.ema_slow)
        confirm_spec = _ResampleSpec(self.confirm_tf, self.ema_fast, self.ema_slow, self.ema_trend)

        trend_df = _resample_ohlcv(data, trend_spec.rule)
        confirm_df = _resample_ohlcv(data, confirm_spec.rule)

        trend_close = trend_df["close"]
        trend_ema_fast = _ema(trend_close, trend_spec.ema_fast)
        trend_ema_slow = _ema(trend_close, trend_spec.ema_slow)
        trend_vwap = _vwap(trend_df)
        trend_adx = _adx(trend_df, self.adx_window)

        confirm_close = confirm_df["close"]
        confirm_ema_trend = _ema(confirm_close, confirm_spec.ema_trend or self.ema_trend)
        confirm_adx = _adx(confirm_df, self.adx_window)

        base_index = base.index
        trend_ema_fast_aligned = _align(trend_ema_fast, base_index)
        trend_ema_slow_aligned = _align(trend_ema_slow, base_index)
        trend_vwap_aligned = _align(trend_vwap, base_index)
        trend_adx_aligned = _align(trend_adx, base_index)

        confirm_close_aligned = _align(confirm_close, base_index)
        confirm_ema_trend_aligned = _align(confirm_ema_trend, base_index)
        confirm_adx_aligned = _align(confirm_adx, base_index)

        ema_spread = (trend_ema_fast_aligned - trend_ema_slow_aligned).abs() / base_close

        trend_regime = (
            (trend_adx_aligned >= self.adx_trend)
            & (confirm_adx_aligned >= self.adx_trend)
            & (ema_spread >= self.ema_flat_threshold)
        )
        range_regime = (
            (trend_adx_aligned < self.adx_range)
            & (confirm_adx_aligned < self.adx_range)
            & (ema_spread < self.ema_flat_threshold)
        )

        trend_long = (
            trend_regime
            & (confirm_close_aligned > confirm_ema_trend_aligned)
            & (trend_ema_fast_aligned > trend_ema_slow_aligned)
            & (base_close > trend_vwap_aligned)
        )
        trend_short = (
            trend_regime
            & (confirm_close_aligned < confirm_ema_trend_aligned)
            & (trend_ema_fast_aligned < trend_ema_slow_aligned)
            & (base_close < trend_vwap_aligned)
        )

        prev_close = base_close.shift(1)
        prev_high = base_high.shift(1)
        prev_low = base_low.shift(1)
        pullback_long = (
            (prev_close <= base_vwap) & (base_close > base_vwap)
        ) | ((prev_close <= base_ema) & (base_close > base_ema))
        pullback_short = (
            (prev_close >= base_vwap) & (base_close < base_vwap)
        ) | ((prev_close >= base_ema) & (base_close < base_ema))

        if self.require_reversal:
            bullish = base_close > base_open
            bearish = base_close < base_open
            fail_new_low = base_low >= prev_low
            fail_new_high = base_high <= prev_high
            pullback_long = pullback_long & bullish & fail_new_low
            pullback_short = pullback_short & bearish & fail_new_high

        trend_long_signal = trend_long & pullback_long
        trend_short_signal = trend_short & pullback_short

        band = self.vwap_band_atr * base_atr
        range_long_signal = range_regime & (base_close <= base_vwap - band)
        range_short_signal = range_regime & (base_close >= base_vwap + band)

        reasons = pd.Series("", index=base_index, dtype=object)
        reasons[range_long_signal] = "range_long_vwap"
        reasons[range_short_signal] = "range_short_vwap"
        reasons[trend_long_signal] = "trend_long_pullback"
        reasons[trend_short_signal] = "trend_short_pullback"

        return reasons.reindex(data.index, method="ffill").fillna("")
