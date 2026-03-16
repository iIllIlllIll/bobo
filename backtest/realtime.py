from __future__ import annotations

import asyncio
import copy
import json
import math
import logging
import re
import time
from collections import deque
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtest.binance import BinanceAPIError, BinanceFuturesClient, SymbolFilters
from backtest.engine import BacktestEngine
from backtest.runner import _normalize_cheatkey_params
from backtest.strategy_loader import get_strategy_class


logger = logging.getLogger(__name__)


@dataclass
class LiveOrderResult:
    success: bool
    status: str
    detail: str = ""
    order_id: Optional[int] = None
    order_info: Optional[Dict[str, Any]] = None
    attempts: int = 0
    elapsed_seconds: float = 0.0


def _format_http_error(exc: Exception) -> str:
    resp = getattr(exc, "response", None)
    if resp is None:
        return ""
    status = getattr(resp, "status_code", None)
    body = ""
    try:
        body = resp.text
    except Exception:
        body = ""
    if status is not None and body:
        return f" status={status} body={body}"
    if status is not None:
        return f" status={status}"
    if body:
        return f" body={body}"
    return ""


def _extract_binance_error(exc: Exception) -> tuple[Optional[int], Optional[str], Optional[str]]:
    if isinstance(exc, BinanceAPIError):
        return exc.code, exc.msg, exc.response_text
    resp = getattr(exc, "response", None)
    if resp is None:
        return None, None, None
    code = None
    msg = None
    body = None
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            code = payload.get("code")
            msg = payload.get("msg")
    except Exception:
        pass
    try:
        body = resp.text
    except Exception:
        body = None
    return code, msg, body


def _format_binance_error(exc: Exception) -> str:
    code, msg, body = _extract_binance_error(exc)
    parts = []
    if code is not None:
        parts.append(f"code={code}")
    if msg:
        parts.append(f"msg={msg}")
    if body:
        parts.append(f"body={body}")
    return " ".join(parts)


def _is_post_only_reject(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code in {-5022, -2010}:
        return True
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return "post only" in text or "post-only" in text or "immediately trigger" in text


def _is_reduce_only_not_required(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code != -1106:
        return False
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return "reduceonly" in text and "not required" in text


def _is_reduce_only_rejected(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code == -2022:
        return True
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return "reduceonly" in text and "rejected" in text


def _is_filter_error(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code in {-1013, -4016, -1111, -4164}:
        return True
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return (
        "price_filter" in text
        or "lot_size" in text
        or "min_notional" in text
        or "filter" in text
        or "invalid price" in text
        or "invalid quantity" in text
        or "precision is over the maximum defined" in text
        or "limit price can't be higher" in text
        or "limit price can't be lower" in text
    )


def _is_insufficient_margin(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code == -2019:
        return True
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return "margin is insufficient" in text


def _extract_limit_price_bound(
    exc: Exception,
) -> tuple[Optional[float], Optional[float]]:
    _, msg, body = _extract_binance_error(exc)
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    upper_match = re.search(r"limit price can't be higher than\s*([0-9]*\.?[0-9]+)", text)
    lower_match = re.search(r"limit price can't be lower than\s*([0-9]*\.?[0-9]+)", text)
    upper = float(upper_match.group(1)) if upper_match else None
    lower = float(lower_match.group(1)) if lower_match else None
    return lower, upper

_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def _interval_seconds(interval: str) -> int:
    return _INTERVAL_SECONDS.get(interval, 60)


def _target_price_for_pnl(
    entry_price: float,
    direction: int,
    leverage_value: float,
    pnl_target: float,
) -> Optional[float]:
    if entry_price <= 0:
        return None
    if leverage_value <= 0:
        leverage_value = 1.0
    pnl_unlevered = float(pnl_target) / leverage_value
    if direction > 0:
        return entry_price * (1.0 + pnl_unlevered)
    return entry_price * (1.0 - pnl_unlevered)


def _klines_to_frame(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    data = []
    for row in rows:
        data.append(
            {
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": int(row[6]),
            }
        )
    return pd.DataFrame(data)


def _normalize_timestamp_value(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.timestamp()
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                try:
                    ts = pd.to_datetime(text, utc=True, errors="raise")
                    return ts.timestamp()
                except Exception:
                    return None
    try:
        ts_val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts_val):
        return None
    abs_val = abs(ts_val)
    if abs_val >= 1e18:
        ts_val = ts_val / 1e9
    elif abs_val >= 1e15:
        ts_val = ts_val / 1e6
    elif abs_val >= 1e12:
        ts_val = ts_val / 1e3
    return ts_val


def _timestamp_key(value: Any) -> Any:
    ts_val = _normalize_timestamp_value(value)
    if ts_val is None:
        return ""
    return int(round(ts_val * 1000))


def _trade_key(trade: Dict[str, Any]) -> tuple:
    trade_type = str(trade.get("type", ""))
    direction = int(trade.get("direction", 0))
    timestamp = _timestamp_key(trade.get("timestamp"))
    entry_reason = str(trade.get("entry_reason") or trade.get("reason") or "")
    exit_reason = str(trade.get("exit_reason", ""))
    entry_time = _timestamp_key(trade.get("entry_time"))
    # Exclude price so live candle recalcs don't spam duplicate messages.
    # Exclude qty so live balance recalcs don't re-emit the same trade.
    return (trade_type, direction, timestamp, entry_reason, exit_reason, entry_time)


@dataclass
class RunnerPayload:
    symbol: str
    interval: str
    price: float
    equity: float
    free_balance: float
    positions: Dict[str, Dict[str, Any]]
    updated_at: float
    engine_positions: Optional[Dict[str, Dict[str, Any]]] = None


class RealtimeRunner:
    def __init__(
        self,
        group: str,
        symbol: str,
        interval: str,
        initial_balance: float,
        fee_rate: float,
        leverage: float,
        position_size: float,
        risk_per_trade: Optional[float],
        max_position_fraction_per_side: Optional[float],
        slippage: float,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        poll_interval: float = 30.0,
        signal_delay_seconds: float = 10.0,
        history_limit: int = 1500,
        max_history: Optional[int] = None,
        binance: Optional[BinanceFuturesClient] = None,
        maker_offset_bps: float = 1.0,
        maker_aggressive_ticks: int = 0,
        auto_trade: bool = False,
        sync_exchange_positions: bool = False,
        sync_interval_seconds: float = 10.0,
        sync_history_path: Optional[str] = None,
        exchange_position_authority: bool = False,
        live_order_retry_seconds: float = 20.0,
        live_order_retry_max_seconds: float = 180.0,
        live_order_max_attempts: int = 0,
        live_order_mode_open: Optional[str] = None,
        live_order_mode_close: Optional[str] = None,
        dual_side_position: Optional[bool] = None,
        state_path: Optional[str] = None,
        persist_seen_limit: int = 2000,
        reset_engine_positions: bool = False,
        on_trade: Optional[Callable[[str, Dict[str, Any], pd.DataFrame], None]] = None,
        on_payload: Optional[Callable[[str, RunnerPayload], None]] = None,
        on_order: Optional[Callable[[str, Dict[str, Any], LiveOrderResult], None]] = None,
    ) -> None:
        self.group = group
        self.symbol = symbol.upper()
        self.interval = interval
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.position_size = position_size
        self.risk_per_trade = risk_per_trade
        self.max_position_fraction_per_side = max_position_fraction_per_side
        self.slippage = slippage
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params
        self._planned_entry_parts = self._resolve_strategy_parts()
        self.poll_interval = poll_interval
        self.signal_delay_seconds = signal_delay_seconds
        self.history_limit = history_limit
        self.max_history = max_history
        self.binance = binance
        self.maker_offset_bps = maker_offset_bps
        self.maker_aggressive_ticks = int(maker_aggressive_ticks or 0)
        self.auto_trade = auto_trade
        self.sync_exchange_positions = sync_exchange_positions
        self.sync_interval_seconds = sync_interval_seconds
        self.sync_history_path = sync_history_path
        self.exchange_position_authority = exchange_position_authority
        self.live_order_retry_seconds = live_order_retry_seconds
        self.live_order_retry_max_seconds = live_order_retry_max_seconds
        self.live_order_max_attempts = live_order_max_attempts
        self.live_order_mode_open = self._normalize_live_order_mode(live_order_mode_open)
        self.live_order_mode_close = self._normalize_live_order_mode(live_order_mode_close)
        self._dual_side_position = dual_side_position
        self.state_path = state_path
        self._persist_seen_limit = max(int(persist_seen_limit or 0), 100)
        self.on_trade = on_trade
        self.on_payload = on_payload
        self.on_order = on_order
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._engine_lock = asyncio.Lock()
        self._data = pd.DataFrame()
        self._last_close_time: Optional[int] = None
        self._live_row: Optional[pd.Series] = None
        self._seen_trades: set[tuple] = set()
        self._recent_trade_keys: deque[tuple] = deque(maxlen=self._persist_seen_limit)
        self._emit_after_ts: Optional[float] = None
        self._last_emitted_ts: Optional[float] = None
        self.latest_payload: Optional[RunnerPayload] = None
        self._sizing_equity: Optional[float] = None
        self._symbol_filters: Optional[SymbolFilters] = None
        self._exchange_positions: Optional[Dict[str, Dict[str, Any]]] = None
        self._exchange_balance: Optional[Dict[str, float]] = None
        self._exchange_leverage: Optional[float] = None
        self._last_exchange_sync: Optional[float] = None
        self._last_sync_snapshot: Optional[str] = None
        self._exchange_sync_ready = (not self.sync_exchange_positions) or (self.binance is None)
        self._last_sync_guard_warning: Optional[float] = None
        self._target_cache: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        self._entry_cache: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        self._exchange_entry_fingerprint: Dict[str, Optional[tuple]] = {
            "long": None,
            "short": None,
        }
        self._exchange_entry_seen_at: Dict[str, Optional[float]] = {
            "long": None,
            "short": None,
        }
        self._pending_exchange_exits: Dict[str, Dict[str, Any]] = {}
        self._pending_exit_log_ts: Dict[str, Optional[float]] = {"long": None, "short": None}
        self._slope_exit_log_ts: Dict[str, Optional[float]] = {"long": None, "short": None}
        self._last_dust_close_ts: Dict[str, Optional[float]] = {"long": None, "short": None}
        self._last_position_mismatch: Optional[str] = None
        self._force_flat_sides: set[str] = set()
        self._last_force_flat_log: Dict[str, Optional[float]] = {"long": None, "short": None}
        self._last_force_flat_sync_log: Dict[str, Optional[float]] = {"long": None, "short": None}
        self._last_user_trade_sync: Optional[float] = None
        self._last_user_trade_time_ms: Optional[int] = None
        self._manual_test_sides: set[str] = set()
        self._suppress_signals_before_ts: Optional[float] = None
        if reset_engine_positions:
            self._suppress_signals_before_ts = time.time()
        else:
            self._load_state()

    def _resolve_strategy_parts(self) -> int:
        try:
            strategy_class = get_strategy_class(self.strategy_name)
            params = _normalize_cheatkey_params(self.strategy_params)
            strategy = strategy_class(**params)
            total_parts = getattr(strategy, "total_parts", None)
            divide_value = getattr(strategy, "divide_value", None)
            parts = int(total_parts) if total_parts else (2 ** int(divide_value) if divide_value else 1)
            if parts <= 0:
                parts = 1
            return parts
        except Exception:
            return 1

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        if self.group == "live" and self.auto_trade:
            # Ignore past signals on restart; only process candles that close after boot.
            self._emit_after_ts = time.time()
        elif self.group == "sim":
            # Sim restarts should not re-emit historical trades to Discord.
            self._emit_after_ts = time.time()
        self._stopped = False
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()

    async def apply_external_snapshot(
        self,
        *,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        price: Optional[float] = None,
        equity: Optional[float] = None,
        free_balance: Optional[float] = None,
        positions: Optional[Dict[str, Dict[str, Any]]] = None,
        engine_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        persist: bool = True,
    ) -> RunnerPayload:
        snapshot_positions: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        if isinstance(positions, dict):
            for side in ("long", "short"):
                raw = positions.get(side)
                if isinstance(raw, dict) and raw:
                    snapshot_positions[side] = dict(raw)
        if engine_positions is None:
            engine_positions = copy.deepcopy(snapshot_positions)
        payload = RunnerPayload(
            symbol=str(symbol or self.symbol),
            interval=str(interval or self.interval),
            price=float(price or 0.0),
            equity=float(equity or 0.0),
            free_balance=float(free_balance or 0.0),
            positions=snapshot_positions,
            engine_positions=engine_positions,
            updated_at=time.time(),
        )
        self.latest_payload = payload
        if self.on_payload is not None:
            self.on_payload(self.group, payload)
        if persist and self.state_path:
            await asyncio.to_thread(self._persist_state)
        return payload

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done() and not self._stopped

    def get_chart_data(self, include_live: bool = True) -> pd.DataFrame:
        if self._data.empty:
            return pd.DataFrame()
        data = self._data.drop(columns=["close_time"], errors="ignore").copy()
        if include_live and self._live_row is not None:
            live_df = self._live_row.to_frame().T.drop(columns=["close_time"], errors="ignore")
            data = pd.concat([data, live_df], ignore_index=True)
        return data.reset_index(drop=True)

    async def _run(self) -> None:
        async with self._engine_lock:
            await self._bootstrap_data()
        interval_sec = _interval_seconds(self.interval)
        while not self._stopped:
            try:
                async with self._engine_lock:
                    await self._update_from_feed()
            except Exception:
                await asyncio.sleep(self.poll_interval)
                continue
            await asyncio.sleep(min(self.poll_interval, interval_sec))

    async def _bootstrap_data(self) -> None:
        if self.binance is None:
            return
        if self.auto_trade and self._symbol_filters is None:
            try:
                self._symbol_filters = await asyncio.to_thread(
                    self.binance.get_symbol_filters, self.symbol
                )
            except Exception:
                self._symbol_filters = None
        rows = await asyncio.to_thread(
            self.binance.get_klines,
            self.symbol,
            self.interval,
            self.history_limit,
        )
        df = _klines_to_frame(rows)
        if df.empty:
            return
        df = df.sort_values("timestamp").reset_index(drop=True)
        now_ms = int(time.time() * 1000)
        if int(df.iloc[-1]["close_time"]) > now_ms:
            self._live_row = df.iloc[-1]
            df = df.iloc[:-1].reset_index(drop=True)
        else:
            self._live_row = None
        if not df.empty:
            self._data = df
            self._last_close_time = int(df.iloc[-1]["close_time"])
        interval_sec = _interval_seconds(self.interval)
        if self.group == "live" and self.auto_trade:
            now = time.time()
            if self._emit_after_ts is None or self._emit_after_ts < (now - interval_sec):
                # Ignore past signals on restart; only process candles that close after boot.
                self._emit_after_ts = now
        elif self._emit_after_ts is None:
            self._emit_after_ts = time.time() - interval_sec
        await self._run_engine(emit_trades=False)

    async def _update_from_feed(self, emit_trades: bool = True) -> None:
        if self.binance is None:
            return
        rows = await asyncio.to_thread(
            self.binance.get_klines,
            self.symbol,
            self.interval,
            2,
        )
        df = _klines_to_frame(rows)
        if df.empty:
            return
        df = df.sort_values("timestamp").reset_index(drop=True)
        now_ms = int(time.time() * 1000)
        live_row: Optional[pd.Series] = None
        if int(df.iloc[-1]["close_time"]) > now_ms:
            live_row = df.iloc[-1]
            df = df.iloc[:-1].reset_index(drop=True)

        if not df.empty:
            last_closed = df.iloc[-1]
            close_time = int(last_closed["close_time"])
            if self._last_close_time is None or close_time > self._last_close_time:
                self._last_close_time = close_time
                self._data = pd.concat([self._data, last_closed.to_frame().T], ignore_index=True)
                self._data = self._data.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                if self.max_history and len(self._data) > self.max_history:
                    self._data = self._data.iloc[-self.max_history :].reset_index(drop=True)

        self._live_row = live_row
        await self._run_engine(emit_trades=emit_trades)

    async def refresh_payload(self) -> bool:
        if self.binance is None:
            return False
        async with self._engine_lock:
            if self._data.empty:
                await self._bootstrap_data()
                return not self._data.empty
            await self._update_from_feed(emit_trades=False)
        return True

    def _should_delay_signals(self) -> bool:
        if self.signal_delay_seconds <= 0:
            return False
        if self._last_close_time is None:
            return False
        delay_ms = int(self.signal_delay_seconds * 1000.0)
        if delay_ms <= 0:
            return False
        now_ms = int(time.time() * 1000)
        return now_ms < (self._last_close_time + delay_ms)

    async def _run_engine(self, emit_trades: bool = True) -> None:
        if self._data.empty:
            return
        data = self._data.drop(columns=["close_time"], errors="ignore")
        ignore_last_row_signals = False
        suppress_signals_from_index: Optional[int] = None
        suppress_signals_before_index: Optional[int] = None
        if self._suppress_signals_before_ts is not None:
            cutoff_ms = int(self._suppress_signals_before_ts * 1000.0)
            try:
                close_times = self._data["close_time"].tolist()
            except Exception:
                close_times = []
            for idx in range(len(close_times) - 1, -1, -1):
                try:
                    close_time_val = int(close_times[idx])
                except Exception:
                    continue
                if close_time_val <= cutoff_ms:
                    suppress_signals_before_index = idx
                    break
        if self._live_row is not None:
            live_df = self._live_row.to_frame().T.drop(columns=["close_time"], errors="ignore")
            data = pd.concat([data, live_df], ignore_index=True)
            ignore_last_row_signals = True
        exit_signal_data = data
        ignore_last_row_exit_signals = ignore_last_row_signals
        if self._live_row is not None and self.group == "live":
            # Use only closed bars for live exit signal decisions to match sim behavior.
            exit_signal_data = self._data.drop(columns=["close_time"], errors="ignore")
            ignore_last_row_exit_signals = False
        if self._should_delay_signals():
            last_closed_index = len(self._data) - 1
            if last_closed_index >= 0:
                suppress_signals_from_index = last_closed_index
        strategy_class = get_strategy_class(self.strategy_name)
        params = _normalize_cheatkey_params(self.strategy_params)
        strategy = strategy_class(**params)
        exit_signals = strategy.exit_signals(exit_signal_data)
        if exit_signals is None:
            exit_signals = pd.Series([0] * len(exit_signal_data), index=exit_signal_data.index)
        exit_signals = exit_signals.reindex(exit_signal_data.index).fillna(0).astype(int)
        allow_live_exit_signals = bool(
            ignore_last_row_exit_signals
            and self.group == "live"
            and self.exchange_position_authority
            and getattr(strategy, "slope_exit_mode", None)
        )
        last_exit_signal: Optional[int] = None
        if allow_live_exit_signals and len(exit_signals) > 0:
            try:
                last_exit_signal = int(exit_signals.iloc[-1])
            except (TypeError, ValueError):
                last_exit_signal = 0
        if ignore_last_row_exit_signals and len(exit_signal_data) > 0:
            exit_signals = exit_signals.copy()
            if not allow_live_exit_signals:
                exit_signals.iloc[-1] = 0
        if suppress_signals_from_index is not None and len(exit_signal_data) > 0:
            idx = int(suppress_signals_from_index)
            if idx < 0:
                idx = 0
            if idx < len(exit_signal_data):
                exit_signals = exit_signals.copy()
                exit_signals.iloc[idx:] = 0
        if allow_live_exit_signals and last_exit_signal is not None and len(exit_signals) > 0:
            exit_signals = exit_signals.copy()
            exit_signals.iloc[-1] = last_exit_signal
        engine_initial_balance = await self._resolve_engine_initial_balance()
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_balance=engine_initial_balance,
            fee_rate=self.fee_rate,
            leverage=self.leverage,
            position_size=self.position_size,
            max_position_fraction_per_side=self.max_position_fraction_per_side,
            risk_per_trade=self.risk_per_trade,
            slippage=self.slippage,
        )
        result = await asyncio.to_thread(
            engine.run,
            None,
            False,
            ignore_last_row_signals,
            suppress_signals_from_index,
            suppress_signals_before_index,
        )
        open_positions = result.open_positions or {}
        blocked_target_sides: set[str] = set()
        if self.group == "live" and self.auto_trade and isinstance(result.trades, list):
            for trade in result.trades:
                if str(trade.get("type", "")) != "entry":
                    continue
                direction = int(trade.get("direction", 0) or 0)
                if direction == 0:
                    continue
                desired_position_side = "LONG" if direction > 0 else "SHORT"
                if self._has_open_positions(desired_position_side):
                    side = "long" if direction > 0 else "short"
                    blocked_target_sides.add(side)
        engine_positions = copy.deepcopy(open_positions)
        # Keep target cache updated even when exchange positions are the authority,
        # so TP/SL targets remain available for live exits.
        self._update_target_cache(open_positions, blocked_sides=blocked_target_sides)
        if not (self.group == "live" and self.exchange_position_authority and self.sync_exchange_positions):
            self._update_entry_cache(open_positions)
        exchange_positions: Optional[Dict[str, Dict[str, Any]]] = None
        if self.group == "live" and self.binance and (
            self.sync_exchange_positions or self.exchange_position_authority
        ):
            exchange_positions = await self._sync_exchange_positions(engine_positions)
            if exchange_positions is not None:
                open_positions = exchange_positions
                if self.exchange_position_authority:
                    engine_positions = copy.deepcopy(exchange_positions)
        if emit_trades:
            trades = result.trades
            if exchange_positions is not None:
                trades = self._reconcile_trades_with_exchange(trades, exchange_positions)
                seeded_trades = self._seed_exchange_exit_trades(
                    result, data, exchange_positions, trades
                )
                if seeded_trades:
                    trades = list(trades) + seeded_trades
                flat_exits = self._seed_exchange_flat_exits(
                    result, data, exchange_positions, trades
                )
                if flat_exits:
                    trades = list(trades) + flat_exits
                if self.exchange_position_authority:
                    slope_exit_config = {
                        "mode": getattr(strategy, "slope_exit_mode", None),
                        "cooldown_bars": getattr(strategy, "slope_exit_cooldown_bars", None),
                        "pnl_threshold": getattr(strategy, "slope_exit_pnl_threshold", None),
                        "use_threshold": getattr(strategy, "slope_exit_use_threshold", True),
                        "ignore_threshold_after_add": getattr(
                            strategy, "slope_exit_ignore_threshold_after_add", False
                        ),
                    }
                    signal_exits = self._seed_exchange_signal_exits(
                        exit_signal_data,
                        exchange_positions,
                        trades,
                        exit_signals,
                        slope_exit_config=slope_exit_config,
                    )
                    if signal_exits:
                        trades = list(trades) + signal_exits
                    trades = self._apply_pending_exchange_exits(
                        trades, data, exchange_positions
                    )
            await self._emit_trades(trades, data)
        self._update_payload(
            result,
            data,
            open_positions_override=open_positions,
            engine_positions_override=engine_positions,
        )

    async def _emit_trades(self, trades: List[Dict[str, Any]], data: pd.DataFrame) -> None:
        emitted = False
        state_dirty = False
        latest_ts: Optional[float] = None
        for trade in trades:
            if self._emit_after_ts is not None:
                ts_val = self._trade_timestamp_seconds(trade)
                if ts_val is not None and ts_val <= self._emit_after_ts:
                    continue
            key = _trade_key(trade)
            if key in self._seen_trades:
                continue
            if (
                self.exchange_position_authority
                and self.sync_exchange_positions
                and self._force_flat_sides
            ):
                trade_type = str(trade.get("type", ""))
                if trade_type == "entry":
                    direction = int(trade.get("direction", 0) or 0)
                    if direction != 0:
                        side = "long" if direction > 0 else "short"
                        if side in self._force_flat_sides:
                            now = time.time()
                            last_log = self._last_force_flat_log.get(side)
                            if last_log is None or (now - last_log) >= 60.0:
                                logger.warning(
                                    "Live entry suppressed to match exchange flat state. "
                                    "symbol=%s side=%s",
                                    self.symbol,
                                    side,
                                )
                                self._last_force_flat_log[side] = now
                            # Mark suppressed entries as seen so we don't spam the same warning.
                            self._seen_trades.add(key)
                            self._recent_trade_keys.append(key)
                            state_dirty = True
                            ts_val = self._trade_timestamp_seconds(trade)
                            if ts_val is not None:
                                latest_ts = ts_val if latest_ts is None else max(latest_ts, ts_val)
                            continue
            ts_val = self._trade_timestamp_seconds(trade)
            if self.auto_trade and self.binance:
                if self._should_block_entry_for_sync(trade):
                    self._log_sync_guard_once()
                    continue
                result = await self._submit_live_order(trade)
                if self.on_order:
                    self.on_order(self.group, trade, result)
                if not result.success:
                    trade_type = str(trade.get("type", ""))
                    # Allow exit trades to retry on the next engine pass if the live order failed.
                    if trade_type in {"exit", "exit_partial"}:
                        if self.exchange_position_authority and self.sync_exchange_positions:
                            self._queue_pending_exchange_exit(trade)
                        continue
                    self._seen_trades.add(key)
                    self._recent_trade_keys.append(key)
                    state_dirty = True
                    if ts_val is not None:
                        latest_ts = ts_val if latest_ts is None else max(latest_ts, ts_val)
                    continue
                trade_type = str(trade.get("type", ""))
                if self.group == "live":
                    self._apply_live_order_timestamps(trade, result)
                    self._stamp_live_trade_times(trade)
                if (
                    trade_type == "entry"
                    and result.order_id is not None
                ):
                    self._update_entry_cache_from_order(trade, result.order_id)
                if (
                    trade_type == "exit"
                    and self.exchange_position_authority
                    and self.sync_exchange_positions
                ):
                    still_open = await self._exchange_position_open_for_trade(
                        trade, force=True
                    )
                    if still_open:
                        logger.warning(
                            "Live exit filled but exchange position still open; will retry. "
                            "symbol=%s direction=%s order_id=%s",
                            self.symbol,
                            int(trade.get("direction", 0) or 0),
                            result.order_id,
                        )
                        self._queue_pending_exchange_exit(trade)
                        continue
            self._seen_trades.add(key)
            self._recent_trade_keys.append(key)
            emitted = True
            state_dirty = True
            if ts_val is not None:
                latest_ts = ts_val if latest_ts is None else max(latest_ts, ts_val)
            if self.on_trade:
                self.on_trade(self.group, trade, data)
            self._update_entry_cache_from_trade(trade)
        if emitted and self.state_path:
            if latest_ts is not None:
                self._last_emitted_ts = latest_ts
                if self._emit_after_ts is None or latest_ts > self._emit_after_ts:
                    self._emit_after_ts = latest_ts
            await asyncio.to_thread(self._persist_state)
        elif state_dirty and self.state_path:
            if latest_ts is not None:
                if self._emit_after_ts is None or latest_ts > self._emit_after_ts:
                    self._emit_after_ts = latest_ts
            await asyncio.to_thread(self._persist_state)

    def _load_state(self) -> None:
        if not self.state_path:
            return
        path = Path(self.state_path)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None
        emit_after = _as_float(payload.get("emit_after_ts"))
        if emit_after is not None:
            self._emit_after_ts = emit_after
        last_emitted = _as_float(payload.get("last_emitted_ts"))
        if last_emitted is not None:
            self._last_emitted_ts = last_emitted
        seen = payload.get("seen_trades")
        if isinstance(seen, list):
            for item in seen:
                if isinstance(item, (list, tuple)):
                    key = tuple(item)
                    if len(key) == 7:
                        key = (key[0], key[1], key[2], key[4], key[5], key[6])
                    if len(key) >= 6:
                        direction_val = _as_float(key[1])
                        direction = int(direction_val) if direction_val is not None else 0
                        normalized = (
                            str(key[0]),
                            direction,
                            _timestamp_key(key[2]),
                            str(key[3]),
                            str(key[4]),
                            _timestamp_key(key[5]),
                        )
                        self._seen_trades.add(normalized)
                        self._recent_trade_keys.append(normalized)
                    else:
                        self._seen_trades.add(key)
                        self._recent_trade_keys.append(key)
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            try:
                    self.latest_payload = RunnerPayload(
                    symbol=str(snapshot.get("symbol", self.symbol)),
                    interval=str(snapshot.get("interval", self.interval)),
                    price=float(snapshot.get("price", 0.0) or 0.0),
                    equity=float(snapshot.get("equity", 0.0) or 0.0),
                    free_balance=float(snapshot.get("free_balance", 0.0) or 0.0),
                    positions=dict(snapshot.get("positions") or {}),
                    engine_positions=dict(snapshot.get("engine_positions") or {})
                    if isinstance(snapshot.get("engine_positions"), dict)
                    else None,
                    updated_at=float(snapshot.get("updated_at", time.time()) or time.time()),
                )
            except Exception:
                self.latest_payload = None
        target_cache = payload.get("target_cache")
        if isinstance(target_cache, dict):
            loaded_cache: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
            for side in ("long", "short"):
                raw = target_cache.get(side)
                if not isinstance(raw, dict) or not raw:
                    continue
                entry_price = _as_float(raw.get("entry_price"))
                if entry_price is None or entry_price <= 0:
                    continue
                cached: Dict[str, Any] = {"entry_price": entry_price}
                updated_at = _as_float(raw.get("updated_at"))
                if updated_at is not None:
                    cached["updated_at"] = updated_at
                for key in (
                    "rr_tp_price",
                    "tp_price",
                    "rr_stop_price",
                    "stop_price",
                    "sl_price",
                    "prev_extreme_stop_price",
                ):
                    value = _as_float(raw.get(key))
                    if value is not None:
                        cached[key] = value
                if bool(raw.get("manual_test_targets")):
                    cached["manual_test_targets"] = True
                if len(cached) > 1:
                    loaded_cache[side] = cached
            if loaded_cache["long"] or loaded_cache["short"]:
                self._target_cache = loaded_cache
        entry_cache = payload.get("entry_cache")
        if isinstance(entry_cache, dict):
            loaded_entries: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
            for side in ("long", "short"):
                raw = entry_cache.get(side)
                if not isinstance(raw, dict) or not raw:
                    continue
                entry_price = _as_float(raw.get("entry_price"))
                if entry_price is None or entry_price <= 0:
                    continue
                cached: Dict[str, Any] = {"entry_price": entry_price}
                updated_at = _as_float(raw.get("updated_at"))
                if updated_at is not None:
                    cached["updated_at"] = updated_at
                try:
                    leverage = float(raw.get("leverage"))
                except (TypeError, ValueError):
                    leverage = None
                if leverage is not None and math.isfinite(leverage):
                    cached["leverage"] = leverage
                try:
                    qty = float(raw.get("qty"))
                except (TypeError, ValueError):
                    qty = None
                if qty is not None and math.isfinite(qty):
                    cached["qty"] = qty
                direction = raw.get("direction")
                if isinstance(direction, (int, float)) and int(direction) != 0:
                    cached["direction"] = int(direction)
                entry_time = raw.get("entry_time")
                if entry_time not in (None, ""):
                    cached["entry_time"] = entry_time
                entry_reason = raw.get("entry_reason")
                if entry_reason not in (None, ""):
                    cached["entry_reason"] = str(entry_reason)
                entry_trade_id = raw.get("entry_trade_id")
                if entry_trade_id not in (None, ""):
                    cached["entry_trade_id"] = entry_trade_id
                entry_order_id = raw.get("entry_order_id")
                if entry_order_id not in (None, ""):
                    cached["entry_order_id"] = entry_order_id
                if len(cached) > 1:
                    loaded_entries[side] = cached
            if loaded_entries["long"] or loaded_entries["short"]:
                self._entry_cache = loaded_entries

    def _persist_state(self) -> None:
        if not self.state_path:
            return
        payload: Dict[str, Any] = {
            "version": 1,
            "group": self.group,
            "symbol": self.symbol,
            "interval": self.interval,
            "emit_after_ts": self._emit_after_ts,
            "last_emitted_ts": self._last_emitted_ts,
            "seen_trades": [list(item) for item in self._recent_trade_keys],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        target_cache: Dict[str, Dict[str, Any]] = {}
        for side in ("long", "short"):
            raw = self._target_cache.get(side)
            if not isinstance(raw, dict) or not raw:
                continue
            try:
                entry_price = float(raw.get("entry_price"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(entry_price) or entry_price <= 0:
                continue
            cached: Dict[str, Any] = {"entry_price": entry_price}
            try:
                updated_at = float(raw.get("updated_at"))
            except (TypeError, ValueError):
                updated_at = None
            if updated_at is not None and math.isfinite(updated_at):
                cached["updated_at"] = updated_at
            for key in (
                "rr_tp_price",
                "tp_price",
                "rr_stop_price",
                "stop_price",
                "sl_price",
                "prev_extreme_stop_price",
            ):
                try:
                    value = float(raw.get(key))
                except (TypeError, ValueError):
                    value = None
                if value is not None and math.isfinite(value):
                    cached[key] = value
            if bool(raw.get("manual_test_targets")):
                cached["manual_test_targets"] = True
            if len(cached) > 1:
                target_cache[side] = cached
        if target_cache:
            payload["target_cache"] = target_cache
        entry_cache: Dict[str, Dict[str, Any]] = {}
        for side in ("long", "short"):
            raw = self._entry_cache.get(side)
            if not isinstance(raw, dict) or not raw:
                continue
            try:
                entry_price = float(raw.get("entry_price"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(entry_price) or entry_price <= 0:
                continue
            cached: Dict[str, Any] = {"entry_price": entry_price}
            try:
                updated_at = float(raw.get("updated_at"))
            except (TypeError, ValueError):
                updated_at = None
            if updated_at is not None and math.isfinite(updated_at):
                cached["updated_at"] = updated_at
            for key in ("leverage", "qty"):
                try:
                    value = float(raw.get(key))
                except (TypeError, ValueError):
                    value = None
                if value is not None and math.isfinite(value):
                    cached[key] = value
            for key in ("target_qty", "part_qty"):
                try:
                    value = float(raw.get(key))
                except (TypeError, ValueError):
                    value = None
                if value is not None and math.isfinite(value):
                    cached[key] = value
            for key in ("base_parts", "parts_filled"):
                try:
                    value = int(raw.get(key))
                except (TypeError, ValueError):
                    value = None
                if value is not None and value >= 0:
                    cached[key] = value
            direction = raw.get("direction")
            if isinstance(direction, (int, float)) and int(direction) != 0:
                cached["direction"] = int(direction)
            entry_time = raw.get("entry_time")
            if entry_time not in (None, ""):
                cached["entry_time"] = entry_time
            entry_reason = raw.get("entry_reason")
            if entry_reason not in (None, ""):
                cached["entry_reason"] = str(entry_reason)
            entry_trade_id = raw.get("entry_trade_id")
            if entry_trade_id not in (None, ""):
                cached["entry_trade_id"] = entry_trade_id
            entry_order_id = raw.get("entry_order_id")
            if entry_order_id not in (None, ""):
                cached["entry_order_id"] = entry_order_id
            if len(cached) > 1:
                entry_cache[side] = cached
        if entry_cache:
            payload["entry_cache"] = entry_cache
        if self.latest_payload is not None:
            payload["snapshot"] = {
                "symbol": self.latest_payload.symbol,
                "interval": self.latest_payload.interval,
                "price": self.latest_payload.price,
                "equity": self.latest_payload.equity,
                "free_balance": self.latest_payload.free_balance,
                "positions": self.latest_payload.positions,
                "engine_positions": self.latest_payload.engine_positions,
                "updated_at": self.latest_payload.updated_at,
            }
        path = Path(self.state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            tmp_path.replace(path)
        except Exception:
            return

    @staticmethod
    def _trade_timestamp_seconds(trade: Dict[str, Any]) -> Optional[float]:
        value = trade.get("timestamp")
        if value is None:
            value = trade.get("exit_time") or trade.get("entry_time")
        return _normalize_timestamp_value(value)

    def _stamp_live_trade_times(self, trade: Dict[str, Any]) -> None:
        trade_type = str(trade.get("type", "")).lower()
        if trade_type not in {"entry", "add", "exit", "exit_partial"}:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        if trade_type in {"entry", "add"}:
            if trade.get("entry_time") in (None, ""):
                trade["entry_time"] = now_iso
            trade.setdefault("timestamp", trade.get("entry_time") or now_iso)
            return
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            if trade.get("exit_time") in (None, ""):
                trade["exit_time"] = now_iso
            trade.setdefault("timestamp", trade.get("exit_time") or now_iso)
            return
        side = "long" if direction > 0 else "short"
        cache = self._entry_cache.get(side) or {}
        cached_entry_time = cache.get("entry_time")
        if cached_entry_time not in (None, "") and trade.get("entry_time") in (None, ""):
            trade["entry_time"] = cached_entry_time
        if trade.get("exit_time") in (None, ""):
            trade["exit_time"] = now_iso
        trade.setdefault("timestamp", trade.get("exit_time") or now_iso)

    def _apply_live_order_timestamps(
        self, trade: Dict[str, Any], result: LiveOrderResult
    ) -> None:
        if result is None or not result.order_info:
            return
        if not isinstance(result.order_info, dict):
            return
        trade_type = str(trade.get("type", "")).lower()
        if trade_type not in {"entry", "add", "exit", "exit_partial"}:
            return
        raw_ts = (
            result.order_info.get("updateTime")
            or result.order_info.get("time")
            or result.order_info.get("transactTime")
        )
        ts_iso = self._timestamp_to_iso(raw_ts)
        if not ts_iso:
            return
        trade["timestamp"] = ts_iso
        if trade_type in {"entry", "add"}:
            trade["entry_time"] = ts_iso
        else:
            trade["exit_time"] = ts_iso

    def _update_payload(
        self,
        result: Any,
        data: pd.DataFrame,
        open_positions_override: Optional[Dict[str, Dict[str, Any]]] = None,
        engine_positions_override: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if data.empty or result is None:
            return
        last_row = data.iloc[-1]
        price = float(last_row["close"])
        equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else 0.0
        free_balance = float(result.final_cash)
        if self.group == "live" and self.sync_exchange_positions and self._exchange_balance:
            available = self._exchange_balance.get("available_balance")
            if available is not None:
                free_balance = available
            exchange_equity = self._exchange_balance.get("equity")
            if exchange_equity is not None:
                equity = exchange_equity
        open_positions = open_positions_override
        if open_positions is None:
            open_positions = result.open_positions or {}
        if self.group == "live":
            open_positions = self._apply_target_cache(open_positions)
        positions = self._build_position_payloads(open_positions, price, equity, free_balance)
        engine_positions_payload: Optional[Dict[str, Dict[str, Any]]] = None
        if (
            self.group == "live"
            and self.sync_exchange_positions
            and isinstance(open_positions_override, dict)
        ):
            engine_positions_override = open_positions_override
        if isinstance(engine_positions_override, dict):
            engine_positions_payload = self._build_position_payloads(
                engine_positions_override, price, equity, free_balance
            )
        if self.group == "live":
            engine_positions_payload = copy.deepcopy(positions)
        payload = RunnerPayload(
            symbol=self.symbol,
            interval=self.interval,
            price=price,
            equity=equity,
            free_balance=free_balance,
            positions=positions,
            engine_positions=engine_positions_payload,
            updated_at=time.time(),
        )
        self.latest_payload = payload
        if self.on_payload:
            self.on_payload(self.group, payload)

    def _build_position_payloads(
        self,
        open_positions: Dict[str, Dict[str, Any]],
        price: float,
        equity: float,
        free_balance: float,
    ) -> Dict[str, Dict[str, Any]]:
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        total_assets = equity if equity > 0 else free_balance
        leverage = self._resolve_effective_leverage()
        position_size = self.position_size if self.position_size > 0 else 1.0
        planned_target_margin = total_assets * position_size
        planned_target_margin *= 0.99
        if not math.isfinite(planned_target_margin) or planned_target_margin <= 0:
            planned_target_margin = None
        planned_target_notional = None
        if planned_target_margin is not None and planned_target_margin > 0:
            planned_target_notional = planned_target_margin * leverage
        planned_target_qty = None
        if price > 0 and planned_target_notional is not None and planned_target_notional > 0:
            planned_target_qty = planned_target_notional / price
        planned_parts = self._planned_entry_parts if self._planned_entry_parts > 0 else 1
        planned_entry_qty = None
        if planned_target_qty is not None and planned_target_qty > 0:
            planned_entry_qty = planned_target_qty / planned_parts
        planned_entry_margin = None
        if planned_target_margin is not None and planned_target_margin > 0:
            planned_entry_margin = planned_target_margin / planned_parts
        planned_entry_notional = None
        if planned_target_notional is not None and planned_target_notional > 0:
            planned_entry_notional = planned_target_notional / planned_parts
        planned_entry_usdt = planned_entry_margin if self.group == "live" else planned_entry_notional
        planned_entry_target_usdt = (
            planned_target_margin if self.group == "live" else planned_target_notional
        )

        payloads: Dict[str, Dict[str, Any]] = {}
        for side in ("long", "short"):
            raw = open_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                payloads[side] = {
                    "active": False,
                    "planned_entry_qty": planned_entry_qty,
                    "planned_entry_target_qty": planned_target_qty,
                    "planned_entry_usdt": planned_entry_usdt,
                    "planned_entry_target_usdt": planned_entry_target_usdt,
                    "planned_entry_parts": planned_parts,
                }
                continue
            entry_price = float(raw.get("entry_price", 0.0) or 0.0)
            qty = float(raw.get("qty", 0.0) or 0.0)
            leverage = float(raw.get("leverage", 1.0) or 1.0)
            direction = int(raw.get("direction", 0) or 0)
            notional = abs(entry_price * qty) if entry_price and qty else 0.0
            pnl_amount = (price - entry_price) * qty * direction
            pnl_pct = 0.0
            if notional > 0:
                pnl_pct = (pnl_amount / notional) * leverage * 100.0

            tp_price = raw.get("rr_tp_price")
            if tp_price is None:
                tp_price = raw.get("tp_price")
            sl_price = raw.get("rr_stop_price")
            if sl_price is None:
                sl_price = raw.get("stop_price")

            tp_pnl = self._target_pnl(entry_price, tp_price, direction, leverage)
            sl_pnl = self._target_pnl(entry_price, sl_price, direction, leverage)

            target_qty = _as_float(raw.get("target_qty"))
            part_qty = _as_float(raw.get("part_qty"))
            base_parts = raw.get("base_parts")
            parts_filled = raw.get("parts_filled")
            entry_remaining_qty = None
            if target_qty is not None and target_qty > 0:
                entry_remaining_qty = max(target_qty - abs(qty), 0.0)

            payloads[side] = {
                "active": True,
                "qty": qty,
                "notional": notional,
                "leverage": leverage,
                "entry_price": entry_price,
                "entry_time": raw.get("entry_time"),
                "entry_trade_id": raw.get("entry_trade_id"),
                "entry_order_id": raw.get("entry_order_id"),
                "entry_target_qty": target_qty,
                "entry_remaining_qty": entry_remaining_qty,
                "entry_part_qty": part_qty,
                "entry_parts": base_parts,
                "entry_parts_filled": parts_filled,
                "direction": direction,
                "tp_pnl": tp_pnl,
                "sl_pnl": sl_pnl,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "pnl_pct": pnl_pct,
                "pnl_amount": pnl_amount,
                "planned_entry_qty": planned_entry_qty,
                "planned_entry_target_qty": planned_target_qty,
                "planned_entry_usdt": planned_entry_usdt,
                "planned_entry_target_usdt": planned_entry_target_usdt,
                "planned_entry_parts": planned_parts,
            }
        return payloads

    def _calc_planned_entry_notional(self, total_assets: float) -> Optional[float]:
        if total_assets <= 0:
            return None
        leverage = self._resolve_effective_leverage()
        position_size = self.position_size if self.position_size > 0 else 1.0
        notional = total_assets * position_size * leverage
        notional *= 0.99
        if not math.isfinite(notional) or notional <= 0:
            return None
        return notional

    def _calc_planned_entry_qty(self, total_assets: float, price: float) -> Optional[float]:
        if price <= 0 or total_assets <= 0:
            return None
        notional = self._calc_planned_entry_notional(total_assets)
        if notional is None:
            return None
        qty = notional / price
        if not math.isfinite(qty) or qty <= 0:
            return None
        return qty

    @staticmethod
    def _target_pnl(
        entry_price: float,
        target_price: Any,
        direction: int,
        leverage: float,
    ) -> Optional[float]:
        if entry_price <= 0 or target_price is None:
            return None
        try:
            target_price_f = float(target_price)
        except (TypeError, ValueError):
            return None
        if target_price_f <= 0:
            return None
        if leverage <= 0:
            leverage = 1.0
        pnl_unlevered = (target_price_f - entry_price) / entry_price * direction
        return pnl_unlevered * leverage * 100.0

    async def _ensure_dual_side_position(self) -> Optional[bool]:
        if self._dual_side_position is not None:
            return self._dual_side_position
        if self.binance is None:
            return None
        try:
            payload = await asyncio.to_thread(self.binance.get_position_mode)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            self._dual_side_position = bool(payload.get("dualSidePosition"))
            return self._dual_side_position
        exchange_positions = await self._get_exchange_positions(force=True)
        inferred = self._infer_dual_side_from_positions(exchange_positions)
        if inferred is not None:
            self._dual_side_position = inferred
            return inferred
        return None

    @staticmethod
    def _normalize_position_side(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value or "").strip().upper()
        if text in {"LONG", "SHORT"}:
            return text
        return None

    @staticmethod
    def _infer_dual_side_from_positions(
        exchange_positions: Optional[Dict[str, Dict[str, Any]]]
    ) -> Optional[bool]:
        if not isinstance(exchange_positions, dict):
            return None
        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            if not isinstance(raw, dict):
                continue
            pos_side = RealtimeRunner._normalize_position_side(
                raw.get("positionSide") or raw.get("position_side")
            )
            if pos_side in {"LONG", "SHORT"}:
                return True
        if exchange_positions:
            return False
        return None

    async def _get_exchange_positions(
        self, force: bool = False
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        if self.binance is None:
            return None
        if (
            not force
            and isinstance(self._exchange_positions, dict)
            and self._exchange_positions
        ):
            return self._exchange_positions
        try:
            raw_positions = await asyncio.to_thread(
                self.binance.get_position_risk, self.symbol
            )
        except Exception:
            return None
        self._update_exchange_leverage(raw_positions)
        exchange_positions = self._parse_exchange_positions(raw_positions)
        self._exchange_positions = exchange_positions
        self._last_exchange_sync = time.time()
        return exchange_positions

    async def _resolve_exchange_exit_qty(self, direction: int) -> Optional[float]:
        if self.binance is None or direction == 0:
            return None
        side = "long" if direction > 0 else "short"
        raw = None
        if isinstance(self._exchange_positions, dict):
            raw = self._exchange_positions.get(side, {})
        try:
            cached_qty = float(raw.get("qty", 0.0) or 0.0) if isinstance(raw, dict) else 0.0
        except (TypeError, ValueError):
            cached_qty = 0.0
        if cached_qty > 0:
            return cached_qty
        exchange_positions = await self._get_exchange_positions()
        if not exchange_positions:
            return None
        raw = exchange_positions.get(side, {})
        try:
            qty = float(raw.get("qty", 0.0) or 0.0) if isinstance(raw, dict) else 0.0
        except (TypeError, ValueError):
            qty = 0.0
        return qty if qty > 0 else None

    async def _resolve_exchange_exit_direction(self, direction: int) -> int:
        if self.binance is None or direction == 0:
            return direction
        exchange_positions = await self._get_exchange_positions()
        if not exchange_positions:
            return direction

        def _qty_for(side: str) -> float:
            raw = exchange_positions.get(side, {}) if exchange_positions else {}
            try:
                value = float(raw.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            return max(value, 0.0)

        long_qty = _qty_for("long")
        short_qty = _qty_for("short")
        if long_qty > 0 and short_qty <= 0:
            return 1
        if short_qty > 0 and long_qty <= 0:
            return -1
        return direction

    async def _resolve_hedge_exit(
        self,
        direction: int,
        qty: float,
        desired_position_side: Optional[str],
    ) -> Tuple[int, float, Optional[str]]:
        if self.binance is None or direction == 0:
            return direction, qty, desired_position_side
        exchange_positions = await self._get_exchange_positions()
        if not exchange_positions:
            position_side = desired_position_side or ("LONG" if direction > 0 else "SHORT")
            return direction, qty, position_side

        def _qty_for(side: str) -> float:
            raw = exchange_positions.get(side, {}) if exchange_positions else {}
            try:
                value = float(raw.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            return max(value, 0.0)

        long_qty = _qty_for("long")
        short_qty = _qty_for("short")

        if desired_position_side in {"LONG", "SHORT"}:
            resolved_direction = 1 if desired_position_side == "LONG" else -1
            resolved_qty = long_qty if desired_position_side == "LONG" else short_qty
            if resolved_qty > 0:
                qty = resolved_qty
            return resolved_direction, qty, desired_position_side

        if long_qty > 0 and short_qty <= 0:
            return 1, long_qty, "LONG"
        if short_qty > 0 and long_qty <= 0:
            return -1, short_qty, "SHORT"

        position_side = "LONG" if direction > 0 else "SHORT"
        if position_side == "LONG" and long_qty > 0:
            qty = long_qty
        elif position_side == "SHORT" and short_qty > 0:
            qty = short_qty
        return direction, qty, position_side

    async def _exchange_position_open_for_trade(
        self, trade: Dict[str, Any], force: bool = False
    ) -> bool:
        if self.binance is None:
            return False
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            return False
        position_side = self._normalize_position_side(
            trade.get("positionSide") or trade.get("position_side")
        )
        if position_side == "LONG":
            side = "long"
        elif position_side == "SHORT":
            side = "short"
        else:
            side = "long" if direction > 0 else "short"
        exchange_positions = await self._get_exchange_positions(force=force)
        if not exchange_positions:
            return False
        raw = exchange_positions.get(side, {})
        try:
            qty = float(raw.get("qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        return qty > 0

    def _queue_pending_exchange_exit(self, trade: Dict[str, Any]) -> None:
        if not isinstance(trade, dict):
            return
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            return
        desired_position_side = self._normalize_position_side(
            trade.get("positionSide") or trade.get("position_side")
        )
        side = "long" if direction > 0 else "short"
        if (
            self.exchange_position_authority
            and self.sync_exchange_positions
            and isinstance(self._exchange_positions, dict)
            and self._exchange_positions
        ):
            def _qty_for(pos_side: str) -> float:
                raw = self._exchange_positions.get(pos_side, {}) or {}
                try:
                    value = float(raw.get("qty", 0.0) or 0.0)
                except (TypeError, ValueError):
                    value = 0.0
                return max(value, 0.0)

            long_qty = _qty_for("long")
            short_qty = _qty_for("short")
            resolved_direction = direction
            resolved_side = side
            if desired_position_side == "LONG" and long_qty > 0:
                resolved_direction = 1
                resolved_side = "long"
            elif desired_position_side == "SHORT" and short_qty > 0:
                resolved_direction = -1
                resolved_side = "short"
            elif long_qty > 0 and short_qty <= 0:
                resolved_direction = 1
                resolved_side = "long"
            elif short_qty > 0 and long_qty <= 0:
                resolved_direction = -1
                resolved_side = "short"

            if resolved_side != side or resolved_direction != direction:
                logger.warning(
                    "Pending exit side resolved from exchange. symbol=%s requested_side=%s "
                    "resolved_side=%s requested_direction=%s resolved_direction=%s "
                    "desired_position_side=%s",
                    self.symbol,
                    side,
                    resolved_side,
                    direction,
                    resolved_direction,
                    desired_position_side or "n/a",
                )
            side = resolved_side
            direction = resolved_direction
        pending = dict(trade)
        pending["direction"] = direction
        pending["type"] = "exit"
        if pending.get("exit_reason") in (None, ""):
            pending["exit_reason"] = "pending_exchange_exit"
        if pending.get("timestamp") in (None, ""):
            pending["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._pending_exchange_exits[side] = pending

    def _apply_pending_exchange_exits(
        self,
        trades: List[Dict[str, Any]],
        data: pd.DataFrame,
        exchange_positions: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.exchange_position_authority or not self.sync_exchange_positions:
            return trades
        if not self._pending_exchange_exits:
            return trades
        if data.empty or not isinstance(exchange_positions, dict):
            return trades

        exit_sides: set[str] = set()
        for trade in trades or []:
            if str(trade.get("type")) not in {"exit", "exit_partial"}:
                continue
            direction = int(trade.get("direction", 0) or 0)
            if direction == 0:
                continue
            exit_sides.add("long" if direction > 0 else "short")

        last_row = data.iloc[-1]
        try:
            close_price = float(last_row.get("close", 0.0) or 0.0)
        except (TypeError, ValueError):
            close_price = 0.0
        timestamp = last_row.get("timestamp", time.time())

        updated_trades = list(trades)
        for side in ("long", "short"):
            pending = self._pending_exchange_exits.get(side)
            if not pending:
                continue
            raw = exchange_positions.get(side, {})
            try:
                qty = float(raw.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                self._pending_exchange_exits.pop(side, None)
                continue
            if side in exit_sides:
                continue
            trade = dict(pending)
            trade["direction"] = 1 if side == "long" else -1
            trade["qty"] = qty
            trade["price"] = close_price
            trade["timestamp"] = timestamp
            trade["positionSide"] = "LONG" if side == "long" else "SHORT"
            if trade.get("entry_price") in (None, "", 0, 0.0):
                trade["entry_price"] = raw.get("entry_price")
            if trade.get("leverage") in (None, "", 0, 0.0):
                trade["leverage"] = raw.get("leverage", self.leverage)
            if trade.get("entry_order_id") in (None, ""):
                entry_order_id = raw.get("entry_order_id")
                if entry_order_id not in (None, ""):
                    trade["entry_order_id"] = entry_order_id
            now = time.time()
            last_log = self._pending_exit_log_ts.get(side)
            if last_log is None or (now - last_log) >= 60.0:
                logger.warning(
                    "Pending exchange exit retry scheduled. symbol=%s side=%s qty=%.8g "
                    "positionSide=%s entry_order_id=%s",
                    self.symbol,
                    side,
                    qty,
                    trade.get("positionSide"),
                    trade.get("entry_order_id"),
                )
                self._pending_exit_log_ts[side] = now
            updated_trades.append(trade)
        return updated_trades

    async def _submit_dust_close(
        self, side: str, position_side: Optional[str], tick_size: float
    ) -> LiveOrderResult:
        if self.binance is None:
            return LiveOrderResult(
                success=False,
                status="skipped",
                detail="binance client not configured",
            )
        try:
            reference_price = await asyncio.to_thread(
                self.binance.get_price, self.symbol
            )
        except Exception as exc:
            detail = _format_binance_error(exc)
            logger.warning(
                "Dust close failed: price fetch error. symbol=%s side=%s error=%s",
                self.symbol,
                side,
                detail or str(exc),
            )
            return LiveOrderResult(
                success=False,
                status="error",
                detail=detail or str(exc),
            )
        # Use a stop price that triggers immediately at the current price.
        stop_price = self._round_price(float(reference_price), tick_size, side)
        if stop_price <= 0:
            stop_price = float(reference_price)
        try:
            order = await asyncio.to_thread(
                self.binance.place_stop_market_close,
                self.symbol,
                side,
                self._format_by_step(stop_price, tick_size),
                True,
                position_side,
            )
        except Exception as exc:
            detail = _format_binance_error(exc)
            logger.warning(
                "Dust close failed: symbol=%s side=%s stop_price=%.8g error=%s",
                self.symbol,
                side,
                stop_price,
                detail or str(exc),
            )
            return LiveOrderResult(
                success=False,
                status="error",
                detail=detail or str(exc),
            )
        order_id = order.get("orderId") if isinstance(order, dict) else None
        return LiveOrderResult(
            success=True,
            status="submitted",
            detail="dust close stop-market submitted",
            order_id=order_id,
        )

    async def _submit_live_order(self, trade: Dict[str, Any]) -> LiveOrderResult:
        start_ts = time.time()
        attempt = 0
        last_emitted_status: Optional[tuple] = None

        def _make_result(
            success: bool,
            status: str,
            detail: str = "",
            order_id: Optional[int] = None,
            order_info: Optional[Dict[str, Any]] = None,
        ) -> LiveOrderResult:
            return LiveOrderResult(
                success=success,
                status=status,
                detail=detail,
                order_id=order_id,
                order_info=order_info,
                attempts=attempt,
                elapsed_seconds=time.time() - start_ts,
            )

        def _emit_event(
            status: str,
            detail: str = "",
            order_id: Optional[int] = None,
        ) -> None:
            nonlocal last_emitted_status
            if self.on_order is None:
                return
            event_key = (status, attempt)
            if event_key == last_emitted_status:
                return
            last_emitted_status = event_key
            result = LiveOrderResult(
                success=False,
                status=status,
                detail=detail,
                order_id=order_id,
                attempts=attempt,
                elapsed_seconds=time.time() - start_ts,
            )
            self.on_order(self.group, trade, result)

        trade_type = trade.get("type")
        if trade_type not in {"entry", "exit", "exit_partial"}:
            return _make_result(False, "skipped", "unsupported trade type")
        if self.binance is None:
            return _make_result(False, "skipped", "binance client not configured")
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            return _make_result(False, "skipped", "missing trade direction")
        qty = float(trade.get("qty", 0.0) or 0.0)
        if qty <= 0:
            return _make_result(False, "skipped", "invalid trade quantity")
        price = float(trade.get("price", 0.0) or 0.0)
        if price <= 0:
            return _make_result(False, "skipped", "invalid trade price")
        is_manual_test = False
        if trade_type == "entry" and self.group == "live":
            entry_reason = str(trade.get("entry_reason", "") or "").lower()
            is_manual_test = entry_reason in {"manual_test", "manual"}
        if trade_type == "entry" and self.group == "live":
            total_assets: Optional[float] = None
            if self.latest_payload is not None:
                payload_equity = float(self.latest_payload.equity or 0.0)
                payload_free = float(self.latest_payload.free_balance or 0.0)
                total_assets = payload_equity if payload_equity > 0 else payload_free
            if total_assets is None or total_assets <= 0:
                if self._exchange_balance:
                    exch_equity = self._exchange_balance.get("equity")
                    exch_free = self._exchange_balance.get("available_balance")
                    try:
                        exch_equity_f = float(exch_equity) if exch_equity is not None else None
                    except (TypeError, ValueError):
                        exch_equity_f = None
                    try:
                        exch_free_f = float(exch_free) if exch_free is not None else None
                    except (TypeError, ValueError):
                        exch_free_f = None
                    if exch_equity_f is not None and exch_equity_f > 0:
                        total_assets = exch_equity_f
                    elif exch_free_f is not None and exch_free_f > 0:
                        total_assets = exch_free_f
            if total_assets is not None and total_assets > 0:
                leverage = self._resolve_effective_leverage(trade.get("leverage"))
                position_size = self.position_size if self.position_size > 0 else 1.0
                planned_target_margin = total_assets * position_size
                planned_target_margin *= 0.99
                planned_target_notional = None
                if math.isfinite(planned_target_margin) and planned_target_margin > 0:
                    planned_target_notional = planned_target_margin * leverage
                planned_parts = self._planned_entry_parts if self._planned_entry_parts > 0 else 1
                if planned_target_notional is not None and planned_target_notional > 0:
                    planned_entry_notional = planned_target_notional / planned_parts
                    planned_entry_margin = planned_target_margin / planned_parts
                    if price > 0 and planned_entry_notional > 0:
                        planned_qty = planned_entry_notional / price
                        if math.isfinite(planned_qty) and planned_qty > 0:
                            qty = planned_qty
                            trade["qty"] = planned_qty
                            trade["planned_entry_usdt"] = planned_entry_margin
                            trade["leverage"] = leverage
                            logger.info(
                                "Live entry sizing aligned to planned entry margin. "
                                "symbol=%s planned_entry_margin=%.8g planned_entry_notional=%.8g "
                                "planned_parts=%s leverage=%.8g price=%.8g qty=%.8g",
                                self.symbol,
                                planned_entry_margin,
                                planned_entry_notional,
                                planned_parts,
                                leverage,
                                price,
                                planned_qty,
                            )
        leverage_val = self._resolve_effective_leverage(trade.get("leverage"))

        dual_side = await self._ensure_dual_side_position()
        dual_side_effective = dual_side
        desired_position_side: Optional[str] = None
        resolved_position_side: Optional[str] = None
        if trade_type == "entry":
            desired_position_side = self._normalize_position_side(
                trade.get("positionSide") or trade.get("position_side")
            )
            if desired_position_side is None:
                desired_position_side = "LONG" if direction > 0 else "SHORT"
            if self.group == "live":
                if self._has_open_positions(desired_position_side):
                    return _make_result(
                        False,
                        "blocked",
                        "entry blocked: same-side position already open",
                    )
                try:
                    payload = await asyncio.to_thread(
                        self.binance.get_position_risk, self.symbol
                    )
                except Exception as exc:
                    detail = _format_binance_error(exc) or str(exc)
                    logger.warning(
                        "Live entry blocked: exchange position check failed. symbol=%s error=%s",
                        self.symbol,
                        detail,
                    )
                    return _make_result(
                        False,
                        "blocked",
                        f"exchange position check failed: {detail}",
                    )
                if self._exchange_has_open_position_side(
                    payload, self.symbol, desired_position_side
                ):
                    logger.warning(
                        "Live entry blocked: same-side exchange position already open. symbol=%s side=%s",
                        self.symbol,
                        desired_position_side,
                    )
                    return _make_result(
                        False,
                        "blocked",
                        "entry blocked: same-side position already open",
                    )
        if trade_type in {"exit", "exit_partial"}:
            desired_position_side = self._normalize_position_side(
                trade.get("positionSide") or trade.get("position_side")
            )

            if desired_position_side and dual_side is not True:
                exchange_positions = await self._get_exchange_positions(force=True)
                if isinstance(exchange_positions, dict):
                    inferred = self._infer_dual_side_from_positions(exchange_positions)
                    if inferred is True:
                        dual_side_effective = True

            if dual_side_effective:
                resolved_direction, resolved_qty, resolved_position_side = (
                    await self._resolve_hedge_exit(
                        direction, qty, desired_position_side
                    )
                )
                if resolved_direction != direction:
                    logger.warning(
                        "Live hedge exit direction overridden by exchange position. "
                        "symbol=%s requested_direction=%s resolved_direction=%s "
                        "desired_position_side=%s",
                        self.symbol,
                        direction,
                        resolved_direction,
                        desired_position_side or "n/a",
                    )
                    direction = resolved_direction
                if resolved_qty is not None and resolved_qty > 0:
                    qty = resolved_qty
                if (
                    desired_position_side is not None
                    and resolved_position_side is not None
                    and resolved_position_side != desired_position_side
                ):
                    logger.warning(
                        "Live hedge exit positionSide overridden by exchange position. "
                        "symbol=%s requested_position_side=%s resolved_position_side=%s",
                        self.symbol,
                        desired_position_side,
                        resolved_position_side,
                    )
            else:
                resolved_direction = await self._resolve_exchange_exit_direction(direction)
                if resolved_direction != direction:
                    logger.warning(
                        "Live exit direction overridden by exchange position. symbol=%s "
                        "requested_direction=%s resolved_direction=%s",
                        self.symbol,
                        direction,
                        resolved_direction,
                    )
                    direction = resolved_direction
                exchange_qty = await self._resolve_exchange_exit_qty(direction)
                if exchange_qty is not None and exchange_qty > 0:
                    qty = exchange_qty

        side = "BUY" if direction > 0 else "SELL"
        reduce_only = False
        position_side = None
        if trade_type in {"exit", "exit_partial"}:
            reduce_only = True
            side = "SELL" if direction > 0 else "BUY"
        if trade_type == "entry" and dual_side_effective:
            position_side = desired_position_side or ("LONG" if direction > 0 else "SHORT")
        if trade_type in {"exit", "exit_partial"}:
            dual_side_for_order = bool(dual_side_effective)
            if (
                not dual_side_for_order
                and desired_position_side
                and dual_side is not True
                and resolved_position_side is not None
            ):
                dual_side_for_order = True
            if dual_side_for_order:
                position_side = resolved_position_side or ("LONG" if direction > 0 else "SHORT")
        checked = await self._ensure_live_leverage(leverage_val)
        if checked is not None and checked > 0:
            leverage_val = checked

        if self._symbol_filters is None:
            try:
                self._symbol_filters = await asyncio.to_thread(
                    self.binance.get_symbol_filters, self.symbol
                )
            except Exception:
                self._symbol_filters = None
        min_notional = (
            self._symbol_filters.min_notional if self._symbol_filters else None
        )

        tick_size = self._symbol_filters.tick_size if self._symbol_filters else 0.0
        step_size = self._symbol_filters.step_size if self._symbol_filters else 0.0
        raw_qty = qty
        remaining_qty = self._round_step_with_tolerance(qty, step_size)
        if remaining_qty <= 0:
            if reduce_only and raw_qty > 0:
                return await self._submit_dust_close(side, position_side, tick_size)
            return _make_result(False, "blocked", "order quantity below step size")

        post_only_rejects = 0
        filters_refreshed = False
        forced_min_price: Optional[float] = None
        forced_max_price: Optional[float] = None
        reduce_only_param = reduce_only
        reduce_only_checks = reduce_only
        reduce_only_omitted = False
        reduce_only_retry_refreshed = False
        cached_available: Optional[float] = None
        cached_equity: Optional[float] = None
        balance_source: Optional[str] = None
        exit_reason = str(trade.get("exit_reason", "") or "").lower()
        is_stop_exit = trade_type in {"exit", "exit_partial"} and (
            "stop" in exit_reason or "sl" in exit_reason
        )
        is_tp_exit = trade_type in {"exit", "exit_partial"} and (
            "take_profit" in exit_reason or "tp" in exit_reason
        )
        sl_price = (
            trade.get("rr_stop_price")
            or trade.get("stop_price")
            or trade.get("sl_price")
        )
        tp_price = trade.get("rr_tp_price") or trade.get("tp_price")

        if trade_type in {"exit", "exit_partial"}:
            # User-requested: always omit reduceOnly on live exits.
            reduce_only_param = False
            reduce_only_omitted = True

        while remaining_qty > 0:
            elapsed = time.time() - start_ts
            force_taker_after_max = (
                reduce_only_checks
                and self.live_order_retry_max_seconds > 0
                and elapsed >= self.live_order_retry_max_seconds
            )
            if (
                self.live_order_retry_max_seconds > 0
                and elapsed >= self.live_order_retry_max_seconds
                and not force_taker_after_max
            ):
                return _make_result(
                    False,
                    "not_filled",
                    f"order retry limit reached after {elapsed:.2f}s",
                )
            attempt += 1
            if self.live_order_max_attempts > 0 and attempt > self.live_order_max_attempts:
                return _make_result(False, "not_filled", "order max attempts reached")

            target_price = price
            reference_price = price
            if attempt > 1:
                try:
                    target_price = await asyncio.to_thread(
                        self.binance.get_price, self.symbol
                    )
                    reference_price = target_price
                except Exception:
                    target_price = price
                    reference_price = target_price
            else:
                try:
                    reference_price = await asyncio.to_thread(
                        self.binance.get_price, self.symbol
                    )
                except Exception:
                    reference_price = target_price
            force_maker_near_market = False
            if is_stop_exit and sl_price is not None:
                try:
                    sl_price_f = float(sl_price)
                except (TypeError, ValueError):
                    sl_price_f = None
                if sl_price_f is not None and sl_price_f > 0 and reference_price > 0:
                    if (direction > 0 and reference_price <= sl_price_f) or (
                        direction < 0 and reference_price >= sl_price_f
                    ):
                        target_price = reference_price
                        force_maker_near_market = True
            if is_tp_exit and tp_price is not None and not force_maker_near_market:
                try:
                    tp_price_f = float(tp_price)
                except (TypeError, ValueError):
                    tp_price_f = None
                if tp_price_f is not None and tp_price_f > 0 and reference_price > 0:
                    if (direction > 0 and reference_price >= tp_price_f) or (
                        direction < 0 and reference_price <= tp_price_f
                    ):
                        target_price = reference_price
                        force_maker_near_market = True

            order_mode = self._resolve_live_order_mode(
                side,
                trade_type in {"exit", "exit_partial"},
            )
            allow_post_only = order_mode == "maker"
            use_post_only = allow_post_only and not force_taker_after_max
            adjusted_price = float(target_price)
            if use_post_only:
                if force_maker_near_market:
                    adjusted_price = self._round_price(
                        float(reference_price), tick_size, side
                    )
                else:
                    adjusted_price = self._maker_price(float(target_price), side)
                    adjusted_price = self._apply_maker_aggressive_ticks(
                        adjusted_price, side, tick_size
                    )
                    adjusted_price = self._round_price(adjusted_price, tick_size, side)
                if post_only_rejects > 0:
                    nudge = tick_size if tick_size > 0 else adjusted_price * 0.0001
                    if nudge > 0:
                        adjusted_price = (
                            adjusted_price - (nudge * post_only_rejects)
                            if side.upper() == "BUY"
                            else adjusted_price + (nudge * post_only_rejects)
                        )
                        adjusted_price = self._round_price(adjusted_price, tick_size, side)
            else:
                adjusted_price = float(reference_price or target_price)
            adjusted_price = self._round_price_with_bounds(
                adjusted_price,
                tick_size,
                side,
                *self._merge_price_bounds(
                    self._price_bounds(float(reference_price), self._symbol_filters),
                    forced_min_price,
                    forced_max_price,
                ),
                marketable=not use_post_only,
            )
            if adjusted_price <= 0:
                return _make_result(False, "error", "invalid adjusted price")
            if min_notional and not reduce_only_checks:
                rounded_notional = remaining_qty * adjusted_price
                if rounded_notional < min_notional:
                    logger.warning(
                        "Live order blocked: notional must be at least %.6g USDT "
                        "(current %.6g USDT). symbol=%s side=%s qty=%.8g price=%.8g",
                        min_notional,
                        rounded_notional,
                        self.symbol,
                        side,
                        remaining_qty,
                        adjusted_price,
                    )
                    return LiveOrderResult(
                        success=False,
                        status="blocked",
                        detail=(
                            f"min_notional={min_notional:.6g} current="
                            f"{rounded_notional:.6g} symbol={self.symbol} side={side} "
                            f"qty={remaining_qty:.8g} price={adjusted_price:.8g}"
                        ),
                    )

            if not reduce_only_checks:
                balance = await self._refresh_exchange_balance()
                if balance:
                    cached_available = balance.get("available_balance")
                    cached_equity = balance.get("equity")
                    balance_source = "fresh"
                elif self._exchange_balance:
                    cached_available = self._exchange_balance.get("available_balance")
                    cached_equity = self._exchange_balance.get("equity")
                    balance_source = "cached"
                if cached_available is not None:
                    required_margin = (remaining_qty * adjusted_price) / leverage_val
                    if required_margin > cached_available:
                        logger.warning(
                            "Live order blocked: insufficient margin. symbol=%s side=%s "
                            "qty=%.8g price=%.8g notional=%.8g leverage=%.6g "
                            "required_margin=%.8g available_balance=%.8g equity=%s "
                            "reduce_only=%s balance_source=%s",
                            self.symbol,
                            side,
                            remaining_qty,
                            adjusted_price,
                            remaining_qty * adjusted_price,
                            leverage_val,
                            required_margin,
                            cached_available,
                            "n/a" if cached_equity is None else f"{cached_equity:.8g}",
                            reduce_only_param,
                            balance_source or "unknown",
                        )
                        await self._log_live_margin_context(
                            side,
                            remaining_qty,
                            adjusted_price,
                            reduce_only_param,
                            (
                                "precheck_insufficient_margin "
                                f"required={required_margin:.8g} "
                                f"available={cached_available:.8g} "
                                f"leverage={leverage_val:.6g}"
                            ),
                        )
                        if self.live_order_retry_seconds > 0:
                            logger.warning(
                                "Live order retry scheduled after insufficient margin "
                                "precheck: retry_in=%.2fs symbol=%s side=%s",
                                self.live_order_retry_seconds,
                                self.symbol,
                                side,
                            )
                            await asyncio.sleep(self.live_order_retry_seconds)
                            cached_available = None
                            cached_equity = None
                            balance_source = None
                            continue
                        return _make_result(
                            False,
                            "blocked",
                            (
                                f"insufficient margin: required={required_margin:.8g} "
                                f"available={cached_available:.8g} leverage={leverage_val:.6g} "
                                f"symbol={self.symbol} side={side} qty={remaining_qty:.8g} "
                                f"price={adjusted_price:.8g}"
                            ),
                        )

            try:
                qty_param = self._format_by_step(remaining_qty, step_size)
                price_param = self._format_by_step(adjusted_price, tick_size)
                if use_post_only:
                    order = await asyncio.to_thread(
                        self.binance.place_limit_maker,
                        self.symbol,
                        side,
                        qty_param,
                        price_param,
                        reduce_only_param,
                        position_side,
                    )
                else:
                    order = await asyncio.to_thread(
                        self.binance.place_market,
                        self.symbol,
                        side,
                        qty_param,
                        reduce_only_param,
                        position_side,
                    )
            except Exception as exc:
                bound_min, bound_max = _extract_limit_price_bound(exc)
                if bound_min is not None or bound_max is not None:
                    if bound_min is not None:
                        forced_min_price = (
                            bound_min
                            if forced_min_price is None
                            else max(forced_min_price, bound_min)
                        )
                    if bound_max is not None:
                        forced_max_price = (
                            bound_max
                            if forced_max_price is None
                            else min(forced_max_price, bound_max)
                        )
                    continue
                if _is_reduce_only_rejected(exc):
                    detail = _format_binance_error(exc)
                    exchange_positions = await self._get_exchange_positions(force=True)
                    await self._log_live_exit_rejection_context(
                        reason="reduce_only_rejected",
                        error_detail=detail or str(exc),
                        side=side,
                        position_side=position_side,
                        desired_position_side=desired_position_side,
                        direction=direction,
                        qty=qty,
                        remaining_qty=remaining_qty,
                        price=adjusted_price,
                        reduce_only=reduce_only_param,
                        use_post_only=use_post_only,
                        leverage=leverage_val,
                        tick_size=tick_size,
                        step_size=step_size,
                        attempt=attempt,
                        exchange_positions=exchange_positions,
                    )
                    ex_qty = 0.0
                    side_key = None
                    if position_side in {"LONG", "SHORT"}:
                        side_key = "long" if position_side == "LONG" else "short"
                    elif direction != 0:
                        side_key = "long" if direction > 0 else "short"
                    if side_key and isinstance(exchange_positions, dict):
                        raw = exchange_positions.get(side_key, {}) or {}
                        try:
                            ex_qty = float(raw.get("qty", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            ex_qty = 0.0
                    if ex_qty <= 0:
                        return _make_result(
                            True,
                            "skipped",
                            "reduceOnly rejected: no open position on exchange",
                        )
                    return _make_result(
                        False,
                        "blocked",
                        "reduceOnly rejected: position side mismatch; exit aborted",
                    )
                if _is_reduce_only_not_required(exc) and reduce_only_param:
                    detail = _format_binance_error(exc)
                    await self._log_live_exit_rejection_context(
                        reason="reduce_only_not_required",
                        error_detail=detail or str(exc),
                        side=side,
                        position_side=position_side,
                        desired_position_side=desired_position_side,
                        direction=direction,
                        qty=qty,
                        remaining_qty=remaining_qty,
                        price=adjusted_price,
                        reduce_only=reduce_only_param,
                        use_post_only=use_post_only,
                        leverage=leverage_val,
                        tick_size=tick_size,
                        step_size=step_size,
                        attempt=attempt,
                        exchange_positions=self._exchange_positions,
                    )
                    if not reduce_only_retry_refreshed:
                        reduce_only_retry_refreshed = True

                        def _resolve_from_exchange_positions(
                            exchange_positions: Dict[str, Dict[str, Any]],
                        ) -> Tuple[int, Optional[float], Optional[str]]:
                            def _qty_for(side_name: str) -> float:
                                raw = exchange_positions.get(side_name, {}) if exchange_positions else {}
                                try:
                                    value = float(raw.get("qty", 0.0) or 0.0)
                                except (TypeError, ValueError):
                                    value = 0.0
                                return max(value, 0.0)

                            long_qty = _qty_for("long")
                            short_qty = _qty_for("short")
                            resolved_direction = direction
                            resolved_qty = None
                            resolved_side = position_side
                            if dual_side:
                                if desired_position_side == "LONG" and long_qty > 0:
                                    resolved_direction = 1
                                    resolved_qty = long_qty
                                    resolved_side = "LONG"
                                elif desired_position_side == "SHORT" and short_qty > 0:
                                    resolved_direction = -1
                                    resolved_qty = short_qty
                                    resolved_side = "SHORT"
                                if resolved_qty is None:
                                    if long_qty > 0 and short_qty <= 0:
                                        resolved_direction = 1
                                        resolved_qty = long_qty
                                        resolved_side = "LONG"
                                    elif short_qty > 0 and long_qty <= 0:
                                        resolved_direction = -1
                                        resolved_qty = short_qty
                                        resolved_side = "SHORT"
                                    elif direction > 0 and long_qty > 0:
                                        resolved_direction = 1
                                        resolved_qty = long_qty
                                        resolved_side = "LONG"
                                    elif direction < 0 and short_qty > 0:
                                        resolved_direction = -1
                                        resolved_qty = short_qty
                                        resolved_side = "SHORT"
                            else:
                                if long_qty > 0 and short_qty <= 0:
                                    resolved_direction = 1
                                    resolved_qty = long_qty
                                elif short_qty > 0 and long_qty <= 0:
                                    resolved_direction = -1
                                    resolved_qty = short_qty
                            return resolved_direction, resolved_qty, resolved_side

                        resolved = False
                        for refresh_idx in range(2):
                            if refresh_idx == 1:
                                await asyncio.sleep(
                                    self.live_order_retry_seconds
                                    if self.live_order_retry_seconds > 0
                                    else 0.2
                                )
                            exchange_positions = await self._get_exchange_positions(force=True)
                            await self._log_live_exit_rejection_context(
                                reason="reduce_only_not_required_refresh",
                                error_detail=detail or str(exc),
                                side=side,
                                position_side=position_side,
                                desired_position_side=desired_position_side,
                                direction=direction,
                                qty=qty,
                                remaining_qty=remaining_qty,
                                price=adjusted_price,
                                reduce_only=reduce_only_param,
                                use_post_only=use_post_only,
                                leverage=leverage_val,
                                tick_size=tick_size,
                                step_size=step_size,
                                attempt=attempt,
                                exchange_positions=exchange_positions,
                                refresh_idx=refresh_idx + 1,
                            )
                            if not exchange_positions:
                                continue
                            resolved_direction, resolved_qty, resolved_side = (
                                _resolve_from_exchange_positions(exchange_positions)
                            )
                            if resolved_qty is None or resolved_qty <= 0:
                                continue
                            direction = resolved_direction
                            qty = resolved_qty
                            side = "SELL" if direction > 0 else "BUY"
                            if dual_side:
                                position_side = resolved_side or ("LONG" if direction > 0 else "SHORT")
                            remaining_qty = self._round_step_with_tolerance(qty, step_size)
                            if remaining_qty <= 0:
                                return _make_result(
                                    False, "blocked", "order quantity below step size"
                                )
                            logger.warning(
                                "Live exit reduceOnly rejected; refreshed exchange position and retrying. "
                                "symbol=%s side=%s positionSide=%s qty=%.8g error=%s",
                                self.symbol,
                                side,
                                position_side or "n/a",
                                remaining_qty,
                                detail or str(exc),
                            )
                            resolved = True
                            break
                        if resolved:
                            # We already refreshed/extracted exchange positions for this error.
                            # Let the no-reduceOnly retry logic below decide the next step.
                            pass
                    if not reduce_only_omitted:
                        exchange_positions = await self._get_exchange_positions(force=True)
                        if not exchange_positions and isinstance(self._exchange_positions, dict):
                            exchange_positions = self._exchange_positions
                        inferred_dual_side = self._infer_dual_side_from_positions(exchange_positions)
                        dual_side_hint = bool(dual_side_effective) or bool(self._dual_side_position) or bool(
                            inferred_dual_side
                        )
                        if position_side in {"LONG", "SHORT"} and isinstance(exchange_positions, dict):
                            side_key = "long" if position_side == "LONG" else "short"
                            raw = exchange_positions.get(side_key, {}) or {}
                            try:
                                exch_qty = float(raw.get("qty", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                exch_qty = 0.0
                            exch_pos_side = self._normalize_position_side(
                                raw.get("positionSide") or raw.get("position_side")
                            )
                            pos_side_ok = exch_pos_side in {None, position_side}
                            if exch_qty > 0 and pos_side_ok and (remaining_qty <= exch_qty or step_size <= 0):
                                reduce_only_param = False
                                reduce_only_omitted = True
                                await self._log_live_exit_rejection_context(
                                    reason="reduce_only_not_required_retry_without_reduce_only",
                                    error_detail=detail or str(exc),
                                    side=side,
                                    position_side=position_side,
                                    desired_position_side=desired_position_side,
                                    direction=direction,
                                    qty=qty,
                                    remaining_qty=remaining_qty,
                                    price=adjusted_price,
                                    reduce_only=reduce_only_param,
                                    use_post_only=use_post_only,
                                    leverage=leverage_val,
                                    tick_size=tick_size,
                                    step_size=step_size,
                                    attempt=attempt,
                                    exchange_positions=exchange_positions,
                                    retry_without_reduce_only=True,
                                )
                                logger.warning(
                                    "Live exit reduceOnly rejected; retrying without reduceOnly "
                                    "in %s mode to close existing position. "
                                    "symbol=%s side=%s positionSide=%s qty=%.8g exch_qty=%.8g",
                                    "hedge" if dual_side_hint else "one-way",
                                    self.symbol,
                                    side,
                                    position_side,
                                    remaining_qty,
                                    exch_qty,
                                )
                                continue
                    logger.warning(
                        "Live exit blocked: reduceOnly not required. "
                        "Refusing to retry without reduceOnly to avoid accidental position flip. "
                        "symbol=%s side=%s qty=%.8g price=%.8g error=%s",
                        self.symbol,
                        side,
                        remaining_qty,
                        adjusted_price,
                        detail or str(exc),
                    )
                    return _make_result(
                        False,
                        "blocked",
                        "reduceOnly not required (no open position or wrong side); "
                        "exit aborted to avoid opening a new position",
                    )
                if _is_filter_error(exc) and not filters_refreshed:
                    filters_refreshed = True
                    try:
                        self._symbol_filters = await asyncio.to_thread(
                            self.binance.get_symbol_filters, self.symbol
                        )
                        if self._symbol_filters:
                            tick_size = self._symbol_filters.tick_size
                            step_size = self._symbol_filters.step_size
                            remaining_qty = self._round_step(remaining_qty, step_size)
                            if remaining_qty <= 0:
                                return _make_result(False, "blocked", "order quantity below step size")
                    except Exception:
                        pass
                    continue
                if _is_post_only_reject(exc):
                    post_only_rejects += 1
                    continue
                detail = _format_binance_error(exc)
                if _is_insufficient_margin(exc):
                    if cached_available is None and not reduce_only_param:
                        balance = await self._refresh_exchange_balance()
                        if balance:
                            cached_available = balance.get("available_balance")
                            cached_equity = balance.get("equity")
                            balance_source = "fresh"
                    required_margin = (remaining_qty * adjusted_price) / leverage_val
                    logger.warning(
                        "Live order failed: insufficient margin. symbol=%s side=%s "
                        "qty=%.8g price=%.8g notional=%.8g leverage=%.6g "
                        "required_margin=%.8g available_balance=%s equity=%s "
                        "reduce_only=%s balance_source=%s error=%s",
                        self.symbol,
                        side,
                        remaining_qty,
                        adjusted_price,
                        remaining_qty * adjusted_price,
                        leverage_val,
                        required_margin,
                        "n/a" if cached_available is None else f"{cached_available:.8g}",
                        "n/a" if cached_equity is None else f"{cached_equity:.8g}",
                        reduce_only_param,
                        balance_source or "unknown",
                        detail or str(exc),
                    )
                    await self._log_live_margin_context(
                        side,
                        remaining_qty,
                        adjusted_price,
                        reduce_only_param,
                        detail or str(exc),
                    )
                    if self.live_order_retry_seconds > 0:
                        logger.warning(
                            "Live order retry scheduled after insufficient margin "
                            "error: retry_in=%.2fs symbol=%s side=%s",
                            self.live_order_retry_seconds,
                            self.symbol,
                            side,
                        )
                        await asyncio.sleep(self.live_order_retry_seconds)
                        cached_available = None
                        cached_equity = None
                        balance_source = None
                        continue
                logger.warning(
                    "Live order failed: symbol=%s side=%s qty=%.8g price=%.8g error=%s",
                    self.symbol,
                    side,
                    remaining_qty,
                    adjusted_price,
                    detail or str(exc),
                )
                return _make_result(False, "error", detail or str(exc))

            order_id = order.get("orderId") if isinstance(order, dict) else None
            _emit_event(
                "submitted",
                (
                    f"order submitted (attempt {attempt}) "
                    f"qty={remaining_qty:.8g} price={adjusted_price:.8g}"
                ),
                order_id,
            )

            skip_fill_check = self.live_order_retry_seconds <= 0
            status_sleep = self.live_order_retry_seconds
            if force_taker_after_max:
                skip_fill_check = False
                status_sleep = 0.0
            if skip_fill_check:
                return _make_result(
                    False,
                    "submitted",
                    "order submitted without fill check",
                    order_id,
                )

            if order_id is None:
                return _make_result(False, "error", "missing order_id")

            if status_sleep > 0:
                await asyncio.sleep(status_sleep)

            status = ""
            executed_qty = 0.0
            try:
                order_status = await asyncio.to_thread(
                    self.binance.get_order, self.symbol, order_id
                )
                if isinstance(order_status, dict):
                    status = str(order_status.get("status", "")).upper()
                    executed_qty = float(order_status.get("executedQty", 0.0) or 0.0)
            except Exception:
                return _make_result(False, "error", "order status check failed", order_id)

            if status == "FILLED":
                self._last_exchange_sync = None
                return _make_result(
                    True,
                    "filled",
                    order_id=order_id,
                    order_info=order_status if isinstance(order_status, dict) else None,
                )

            try:
                await asyncio.to_thread(self.binance.cancel_order, self.symbol, order_id)
            except Exception:
                pass
            _emit_event("canceled", "order canceled after not filled", order_id)

            raw_remaining_qty = max(remaining_qty - executed_qty, 0.0)
            remaining_qty = self._round_step_with_tolerance(raw_remaining_qty, step_size)
            self._last_exchange_sync = None
            if remaining_qty <= 0:
                if reduce_only_param and raw_remaining_qty > 0:
                    return await self._submit_dust_close(
                        side, position_side, tick_size
                    )
                return _make_result(True, "filled", order_id=order_id)
            if force_taker_after_max:
                    return _make_result(
                        False,
                        "not_filled",
                        "market submitted after retry limit but remaining qty unfilled",
                        order_id,
                    )
            if (
                self.live_order_retry_max_seconds > 0
                and (time.time() - start_ts) >= self.live_order_retry_max_seconds
            ):
                elapsed = time.time() - start_ts
                return _make_result(
                    False,
                    "not_filled",
                    f"order retry limit reached after {elapsed:.2f}s",
                    order_id,
                )
            _emit_event(
                "retrying",
                f"retrying order with remaining qty {remaining_qty:.8g}",
                order_id,
            )
        return _make_result(False, "not_filled", "order loop exited without fill")

    async def build_livetest_signal_entry_trade(
        self,
        direction: int,
        price: float = 0.0,
        equity_override: Optional[float] = None,
        leverage_override: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if direction not in (1, -1):
            return None, "Usage: `!livetest buy|sell [equity_override] [price] [leverage]`"

        data = self.get_chart_data(include_live=True)
        if data.empty:
            return None, "No chart data available."

        if price <= 0:
            last_row = data.iloc[-1]
            try:
                price = float(last_row.get("close", 0.0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
        if price <= 0:
            return None, f"Price unavailable for {self.symbol}."

        if leverage_override is None and self.group == "live" and self.binance is not None:
            if self._exchange_leverage is None or self._exchange_leverage <= 0:
                try:
                    risk_payload = await asyncio.to_thread(
                        self.binance.get_position_risk, self.symbol
                    )
                    self._update_exchange_leverage(risk_payload)
                except Exception:
                    pass
        leverage = self._resolve_effective_leverage(leverage_override)

        total_assets: Optional[float] = None
        if equity_override is not None and equity_override > 0:
            total_assets = float(equity_override)
        else:
            total_assets = await self._resolve_engine_initial_balance()
        if total_assets is None or total_assets <= 0:
            return None, "Balance unavailable. Provide `equity_override`, e.g. `!livetest buy 20`."

        position_size = self.position_size if self.position_size > 0 else 1.0
        planned_target_margin = total_assets * position_size
        planned_target_margin *= 0.99
        planned_target_notional = planned_target_margin * leverage
        planned_parts = self._planned_entry_parts if self._planned_entry_parts > 0 else 1
        planned_entry_notional = planned_target_notional / planned_parts
        planned_entry_margin = planned_target_margin / planned_parts
        qty = planned_entry_notional / price if price > 0 else 0.0
        if qty <= 0:
            return None, "Calculated quantity is too small."

        now = datetime.now(timezone.utc)
        trade = {
            "type": "entry",
            "direction": direction,
            "qty": qty,
            "price": price,
            "entry_price": price,
            "leverage": leverage,
            "timestamp": now,
            "entry_time": now,
            "entry_reason": "manual_test",
            "planned_entry_usdt": planned_entry_margin,
            "positionSide": "LONG" if direction > 0 else "SHORT",
        }
        return trade, None

    def _resolve_effective_leverage(self, leverage_override: Any = None) -> float:
        if leverage_override not in (None, "", 0, 0.0):
            try:
                override_val = float(leverage_override)
            except (TypeError, ValueError):
                override_val = 0.0
            if override_val > 0 and math.isfinite(override_val):
                return override_val
        if self.group == "live":
            try:
                live_val = float(self._exchange_leverage or 0.0)
            except (TypeError, ValueError):
                live_val = 0.0
            if live_val > 0 and math.isfinite(live_val):
                return live_val
        try:
            cfg_val = float(self.leverage or 0.0)
        except (TypeError, ValueError):
            cfg_val = 0.0
        if cfg_val > 0 and math.isfinite(cfg_val):
            return cfg_val
        return 1.0

    def _update_exchange_leverage(self, raw_positions: Any) -> None:
        if self.group != "live":
            return
        risk = self._extract_symbol_risk(raw_positions, self.symbol)
        if not isinstance(risk, dict):
            return
        try:
            leverage = float(risk.get("leverage", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        if leverage > 0 and math.isfinite(leverage):
            self._exchange_leverage = leverage

    @staticmethod
    def _merge_price_bounds(
        base_bounds: Tuple[Optional[float], Optional[float]],
        forced_min: Optional[float],
        forced_max: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        min_price, max_price = base_bounds
        if forced_min is not None:
            min_price = forced_min if min_price is None else max(min_price, forced_min)
        if forced_max is not None:
            max_price = forced_max if max_price is None else min(max_price, forced_max)
        return min_price, max_price

    def _should_sync_exchange(self) -> bool:
        if not (self.sync_exchange_positions or self.exchange_position_authority):
            return False
        if self.binance is None:
            return False
        if self.sync_interval_seconds <= 0:
            return True
        if self._last_exchange_sync is None:
            return True
        return (time.time() - self._last_exchange_sync) >= self.sync_interval_seconds

    async def _sync_exchange_positions(
        self, engine_positions: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        if (
            self.exchange_position_authority
            and self._force_flat_sides
            and isinstance(engine_positions, dict)
        ):
            for side in self._force_flat_sides:
                engine_positions[side] = {}
        if not self._should_sync_exchange():
            return self._exchange_positions
        if self.binance is None:
            return self._exchange_positions
        try:
            raw_positions = await asyncio.to_thread(
                self.binance.get_position_risk, self.symbol
            )
        except Exception as exc:
            details = _format_http_error(exc)
            logger.warning(
                "Exchange position sync failed for %s: %s.%s",
                self.symbol,
                exc,
                details,
            )
            self._exchange_sync_ready = False
            return self._exchange_positions
        if self._symbol_filters is None:
            try:
                self._symbol_filters = await asyncio.to_thread(
                    self.binance.get_symbol_filters, self.symbol
                )
            except Exception:
                self._symbol_filters = None
        self._update_exchange_leverage(raw_positions)
        exchange_positions = self._parse_exchange_positions(raw_positions)
        dust_positions = self._extract_dust_positions(raw_positions)
        await self._maybe_submit_dust_closes(dust_positions, engine_positions)
        self._update_entry_cache_from_exchange(exchange_positions)
        await self._update_entry_cache_from_fills(exchange_positions)
        self._exchange_sync_ready = True
        try:
            raw_account = await asyncio.to_thread(self.binance.get_account)
            parsed_account = self._parse_exchange_account(raw_account)
            if parsed_account:
                self._exchange_balance = parsed_account
            else:
                logger.warning("Exchange account payload parsed empty.")
        except Exception as exc:
            details = _format_http_error(exc)
            logger.warning("Exchange account sync failed: %s.%s", exc, details)
        if not (self.group == "live" and self.exchange_position_authority):
            exchange_positions = self._merge_exchange_positions(
                engine_positions, exchange_positions
            )
        exchange_positions = self._apply_entry_cache(exchange_positions)
        # Always apply target cache (TP/SL) so exchange-authority mode can still exit on targets.
        exchange_positions = self._apply_target_cache(exchange_positions)
        manual_sides = self._manual_test_sides_from_positions(exchange_positions)
        self._manual_test_sides = manual_sides
        if manual_sides and isinstance(engine_positions, dict):
            for side in manual_sides:
                ex_raw = exchange_positions.get(side, {})
                if not isinstance(ex_raw, dict) or not ex_raw:
                    continue
                try:
                    ex_qty = float(ex_raw.get("qty", 0.0) or 0.0)
                except (TypeError, ValueError):
                    ex_qty = 0.0
                if ex_qty <= 0:
                    continue
                eng_raw = engine_positions.get(side, {})
                try:
                    eng_qty = (
                        float(eng_raw.get("qty", 0.0) or 0.0)
                        if isinstance(eng_raw, dict)
                        else 0.0
                    )
                except (TypeError, ValueError):
                    eng_qty = 0.0
                if eng_qty <= 0:
                    engine_positions[side] = dict(ex_raw)
        if self.group == "live" and self.exchange_position_authority and isinstance(engine_positions, dict):
            for side in ("long", "short"):
                engine_positions[side] = dict(exchange_positions.get(side) or {})
        if self.exchange_position_authority:
            force_flat_sides: set[str] = set()
            mismatch_notes: List[str] = []
            for side in ("long", "short"):
                ex_raw = exchange_positions.get(side, {})
                eng_raw = engine_positions.get(side, {})
                try:
                    ex_qty = float(ex_raw.get("qty", 0.0) or 0.0) if isinstance(ex_raw, dict) else 0.0
                except (TypeError, ValueError):
                    ex_qty = 0.0
                try:
                    eng_qty = float(eng_raw.get("qty", 0.0) or 0.0) if isinstance(eng_raw, dict) else 0.0
                except (TypeError, ValueError):
                    eng_qty = 0.0
                if eng_qty > 0 and ex_qty <= 0:
                    force_flat_sides.add(side)
                if (ex_qty > 0 and eng_qty <= 0) or (
                    ex_qty <= 0 and eng_qty > 0 and side not in force_flat_sides
                ):
                    ex_order_id = None
                    eng_order_id = None
                    if isinstance(ex_raw, dict):
                        ex_order_id = ex_raw.get("entry_order_id")
                    if isinstance(eng_raw, dict):
                        eng_order_id = eng_raw.get("entry_order_id")
                    mismatch_notes.append(
                        f"{side}:engine_qty={eng_qty:.8g} exchange_qty={ex_qty:.8g} "
                        f"engine_entry_order_id={eng_order_id} exchange_entry_order_id={ex_order_id}"
                    )
            if force_flat_sides:
                for side in force_flat_sides:
                    self._target_cache[side] = {}
                    self._entry_cache[side] = {}
                    self._pending_exchange_exits.pop(side, None)
                    if isinstance(engine_positions, dict):
                        engine_positions[side] = {}
                now = time.time()
                log_sides = []
                for side in force_flat_sides:
                    last_log = self._last_force_flat_sync_log.get(side)
                    if last_log is None or (now - last_log) >= 300.0:
                        log_sides.append(side)
                        self._last_force_flat_sync_log[side] = now
                if log_sides:
                    logger.warning(
                        "Engine position cleared to match exchange (authority enabled). symbol=%s sides=%s",
                        self.symbol,
                        ",".join(sorted(log_sides)),
                    )
            self._force_flat_sides = force_flat_sides
            if mismatch_notes:
                snapshot = "|".join(mismatch_notes)
                if snapshot != self._last_position_mismatch:
                    logger.warning(
                        "Exchange position mismatch detected (authority enabled). symbol=%s details=%s",
                        self.symbol,
                        snapshot,
                    )
                    self._last_position_mismatch = snapshot
            else:
                self._last_position_mismatch = None
        else:
            self._force_flat_sides = set()
        self._exchange_positions = exchange_positions
        self._last_exchange_sync = time.time()
        self._record_position_sync(engine_positions, exchange_positions)
        return exchange_positions

    @staticmethod
    def _manual_test_sides_from_positions(
        positions: Dict[str, Dict[str, Any]]
    ) -> set[str]:
        sides: set[str] = set()
        if not isinstance(positions, dict):
            return sides
        for side in ("long", "short"):
            raw = positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                continue
            try:
                qty = float(raw.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue
            entry_reason = str(raw.get("entry_reason", "") or "").lower()
            if entry_reason in {"manual_test", "manual"}:
                sides.add(side)
        return sides

    def _should_block_entry_for_sync(self, trade: Dict[str, Any]) -> bool:
        if self.group != "live":
            return False
        if not self.sync_exchange_positions:
            return False
        if self.binance is None:
            return False
        trade_type = str(trade.get("type", ""))
        if trade_type != "entry":
            return False
        return not self._exchange_sync_ready

    def _log_sync_guard_once(self) -> None:
        now = time.time()
        if (
            self._last_sync_guard_warning is None
            or (now - self._last_sync_guard_warning) >= 60.0
        ):
            logger.warning(
                "Live entry blocked: exchange position sync not ready yet. symbol=%s",
                self.symbol,
            )
            self._last_sync_guard_warning = now

    def _update_target_cache(
        self,
        engine_positions: Dict[str, Dict[str, Any]],
        blocked_sides: Optional[set[str]] = None,
    ) -> None:
        if not isinstance(engine_positions, dict):
            return
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None
        for side in ("long", "short"):
            if blocked_sides and side in blocked_sides:
                continue
            if (
                self.exchange_position_authority
                and self.sync_exchange_positions
                and side in self._pending_exchange_exits
            ):
                # Avoid overwriting live targets while an exchange-side exit is pending.
                continue
            raw = engine_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                continue
            entry_price = _as_float(raw.get("entry_price"))
            if entry_price is None or entry_price <= 0:
                continue
            cached: Dict[str, Any] = {"entry_price": entry_price, "updated_at": time.time()}
            for key in (
                "rr_tp_price",
                "tp_price",
                "rr_stop_price",
                "stop_price",
                "sl_price",
                "prev_extreme_stop_price",
            ):
                if raw.get(key) is not None:
                    cached[key] = raw.get(key)
            if len(cached) > 2:
                self._target_cache[side] = cached

    def _apply_target_cache(
        self, exchange_positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(exchange_positions, dict):
            return exchange_positions

        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        def _maybe_set(target: Dict[str, Any], key: str, source: Dict[str, Any], *alts: str) -> None:
            if target.get(key) not in (None, "", 0, 0.0):
                return
            for name in (key, *alts):
                value = source.get(name)
                if value not in (None, "", 0, 0.0):
                    target[key] = value
                    return

        def _cache_recent(side: str, max_age_sec: float) -> bool:
            cache = self._target_cache.get(side) or {}
            if not isinstance(cache, dict) or not cache:
                return False
            updated_at = _as_float(cache.get("updated_at"))
            if updated_at is None:
                return False
            return (time.time() - updated_at) <= max_age_sec

        def _entry_cache_open(side: str, max_age_sec: float) -> bool:
            entry = self._entry_cache.get(side) or {}
            if not isinstance(entry, dict) or not entry:
                return False
            entry_price = _as_float(entry.get("entry_price"))
            if entry_price is None or entry_price <= 0:
                return False
            updated_at = _as_float(entry.get("updated_at"))
            if updated_at is None:
                return False
            return (time.time() - updated_at) <= max_age_sec

        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                if not (_cache_recent(side, 300.0) or _entry_cache_open(side, 300.0)):
                    self._target_cache[side] = {}
                continue
            qty_val = _as_float(raw.get("qty", 0.0)) or 0.0
            if qty_val <= 0:
                if not (_cache_recent(side, 300.0) or _entry_cache_open(side, 300.0)):
                    self._target_cache[side] = {}
                continue
            entry_reason = str(raw.get("entry_reason", "") or "").lower()
            cache = self._target_cache.get(side) or {}
            if entry_reason in {"manual_test", "manual"} and not cache.get("manual_test_targets"):
                has_targets = any(
                    cache.get(key) not in (None, "", 0, 0.0)
                    for key in (
                        "rr_tp_price",
                        "tp_price",
                        "rr_stop_price",
                        "stop_price",
                        "sl_price",
                    )
                )
                if has_targets:
                    cache["manual_test_targets"] = True
                    self._target_cache[side] = cache
                else:
                    self._target_cache[side] = {}
                    continue
            if not cache:
                continue
            ex_entry = _as_float(raw.get("entry_price"))
            cache_entry = _as_float(cache.get("entry_price"))
            if ex_entry and cache_entry:
                diff = abs(ex_entry - cache_entry) / ex_entry
                if diff > 0.01:
                    continue
            _maybe_set(raw, "rr_tp_price", cache, "tp_price")
            _maybe_set(raw, "tp_price", cache, "rr_tp_price")
            _maybe_set(raw, "rr_stop_price", cache, "stop_price", "sl_price")
            _maybe_set(raw, "stop_price", cache, "rr_stop_price", "sl_price")
            _maybe_set(raw, "sl_price", cache, "rr_stop_price", "stop_price")
            _maybe_set(raw, "prev_extreme_stop_price", cache)
            _maybe_set(raw, "manual_test_targets", cache)
            exchange_positions[side] = raw
        return exchange_positions

    def _update_entry_cache(self, engine_positions: Dict[str, Dict[str, Any]]) -> None:
        if not isinstance(engine_positions, dict):
            return
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None
        for side in ("long", "short"):
            if (
                self.exchange_position_authority
                and self.sync_exchange_positions
                and side in self._pending_exchange_exits
            ):
                # Keep prior entry cache while the exchange exit is still pending.
                continue
            raw = engine_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                continue
            entry_price = _as_float(raw.get("entry_price"))
            if entry_price is None or entry_price <= 0:
                continue
            cached: Dict[str, Any] = {"entry_price": entry_price, "updated_at": time.time()}
            leverage = _as_float(raw.get("leverage"))
            if leverage is not None:
                cached["leverage"] = leverage
            qty = _as_float(raw.get("qty"))
            if qty is not None:
                cached["qty"] = qty
            target_qty = _as_float(raw.get("target_qty"))
            if target_qty is not None:
                cached["target_qty"] = target_qty
            part_qty = _as_float(raw.get("part_qty"))
            if part_qty is not None:
                cached["part_qty"] = part_qty
            base_parts = raw.get("base_parts")
            if isinstance(base_parts, (int, float)) and int(base_parts) > 0:
                cached["base_parts"] = int(base_parts)
            parts_filled = raw.get("parts_filled")
            if isinstance(parts_filled, (int, float)) and int(parts_filled) >= 0:
                cached["parts_filled"] = int(parts_filled)
            direction = raw.get("direction")
            if isinstance(direction, (int, float)) and int(direction) != 0:
                cached["direction"] = int(direction)
            entry_time = raw.get("entry_time")
            if entry_time not in (None, ""):
                cached["entry_time"] = entry_time
            entry_reason = raw.get("entry_reason")
            if entry_reason not in (None, ""):
                cached["entry_reason"] = str(entry_reason)
            if len(cached) > 1:
                self._entry_cache[side] = cached

    def _update_entry_cache_from_exchange(
        self, exchange_positions: Dict[str, Dict[str, Any]]
    ) -> None:
        if not isinstance(exchange_positions, dict):
            return

        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                self._exchange_entry_fingerprint[side] = None
                self._exchange_entry_seen_at[side] = None
                continue
            qty = _as_float(raw.get("qty"))
            entry_price = _as_float(raw.get("entry_price"))
            direction = raw.get("direction")
            leverage = _as_float(raw.get("leverage"))
            if qty is None or qty <= 0 or entry_price is None or entry_price <= 0:
                self._exchange_entry_fingerprint[side] = None
                self._exchange_entry_seen_at[side] = None
                continue
            if not isinstance(direction, (int, float)) or int(direction) == 0:
                self._exchange_entry_fingerprint[side] = None
                self._exchange_entry_seen_at[side] = None
                continue
            fingerprint = (
                round(entry_price, 8),
                int(direction),
                round(leverage or 1.0, 4),
            )
            if fingerprint != self._exchange_entry_fingerprint.get(side):
                seen_at = self._timestamp_to_epoch(raw.get("exchange_update_time"))
                if seen_at is None:
                    seen_at = time.time()
                self._exchange_entry_seen_at[side] = seen_at
                entry_time = raw.get("entry_time")
                if entry_time in (None, ""):
                    entry_time = self._timestamp_to_iso(raw.get("exchange_update_time"))
                if entry_time in (None, ""):
                    entry_time = self._timestamp_to_iso(seen_at)
                if entry_time in (None, ""):
                    entry_time = datetime.now(timezone.utc).isoformat()
                raw["entry_time"] = entry_time
                raw.setdefault("entry_reason", "exchange_sync")
                cached: Dict[str, Any] = {
                    "entry_price": entry_price,
                    "qty": qty,
                    "direction": int(direction),
                    "updated_at": time.time(),
                    "entry_time": entry_time,
                    "entry_reason": "exchange_sync",
                }
                if leverage is not None:
                    cached["leverage"] = leverage
                self._entry_cache[side] = cached
                self._exchange_entry_fingerprint[side] = fingerprint
            else:
                cache = dict(self._entry_cache.get(side) or {})
                if cache:
                    cache["updated_at"] = time.time()
                    if cache.get("entry_time") in (None, ""):
                        entry_time = raw.get("entry_time")
                        if entry_time in (None, ""):
                            entry_time = self._timestamp_to_iso(raw.get("exchange_update_time"))
                        if entry_time in (None, ""):
                            entry_time = self._timestamp_to_iso(time.time())
                        if entry_time:
                            cache["entry_time"] = entry_time
                    self._entry_cache[side] = cache

    async def _update_entry_cache_from_fills(
        self, exchange_positions: Dict[str, Dict[str, Any]]
    ) -> None:
        if not isinstance(exchange_positions, dict):
            return
        if self.binance is None:
            return

        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        sides_to_update: list[str] = []
        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                continue
            qty = _as_float(raw.get("qty"))
            if qty is None or qty <= 0:
                continue
            cache = self._entry_cache.get(side) or {}
            if cache.get("entry_trade_id") in (None, "") and cache.get("entry_order_id") in (None, ""):
                sides_to_update.append(side)

        if not sides_to_update:
            return

        now = time.time()
        if (
            self._last_user_trade_sync is not None
            and now - self._last_user_trade_sync < max(self.sync_interval_seconds * 0.5, 2.0)
        ):
            return
        self._last_user_trade_sync = now

        earliest_seen: Optional[float] = None
        for side in sides_to_update:
            seen_at = self._exchange_entry_seen_at.get(side)
            if isinstance(seen_at, (int, float)) and math.isfinite(seen_at):
                if earliest_seen is None or seen_at < earliest_seen:
                    earliest_seen = seen_at
        if earliest_seen is None:
            earliest_seen = now - 3600.0
        start_ms = int(max(0.0, (earliest_seen - 300.0) * 1000.0))
        if self._last_user_trade_time_ms is not None:
            overlap = self._last_user_trade_time_ms - 60_000
            if overlap < start_ms:
                start_ms = max(overlap, 0)

        try:
            raw_trades = await asyncio.to_thread(
                self.binance.get_user_trades, self.symbol, start_ms, None, 500
            )
        except Exception as exc:
            logger.debug(
                "Exchange user trades sync failed for %s: %s",
                self.symbol,
                _format_binance_error(exc) or str(exc),
            )
            return

        if not isinstance(raw_trades, list) or not raw_trades:
            return
        trades: list[tuple[int, Dict[str, Any]]] = []
        for item in raw_trades:
            if not isinstance(item, dict):
                continue
            t_ms = self._trade_time_ms(item)
            if t_ms is None:
                continue
            trades.append((t_ms, item))
        if not trades:
            return
        trades.sort(key=lambda row: row[0])
        last_time_ms = trades[-1][0]
        if self._last_user_trade_time_ms is None or last_time_ms > self._last_user_trade_time_ms:
            self._last_user_trade_time_ms = last_time_ms

        for side in sides_to_update:
            raw = exchange_positions.get(side, {})
            seen_at = self._exchange_entry_seen_at.get(side)
            threshold_ms: Optional[int] = None
            if isinstance(seen_at, (int, float)) and math.isfinite(seen_at):
                threshold_ms = int((seen_at - 300.0) * 1000.0)
            cache = self._entry_cache.get(side) or {}
            target_order_id = cache.get("entry_order_id")
            entry_trade: Optional[Dict[str, Any]] = None
            if target_order_id not in (None, ""):
                target_str = str(target_order_id)
                for t_ms, trade in trades:
                    order_id = trade.get("orderId") or trade.get("order_id")
                    if order_id in (None, ""):
                        continue
                    if str(order_id) != target_str:
                        continue
                    if not self._is_entry_trade_for_side(trade, side):
                        continue
                    entry_trade = trade
                    break
            if entry_trade is None:
                for t_ms, trade in trades:
                    if threshold_ms is not None and t_ms < threshold_ms:
                        continue
                    if not self._is_entry_trade_for_side(trade, side):
                        continue
                    entry_trade = trade
                    break
            if entry_trade is None:
                for t_ms, trade in reversed(trades):
                    if self._is_entry_trade_for_side(trade, side):
                        entry_trade = trade
                        break
            if entry_trade is None:
                continue
            self._apply_entry_trade_cache(side, raw, entry_trade)

    @staticmethod
    def _trade_time_ms(trade: Dict[str, Any]) -> Optional[int]:
        for key in ("time", "T", "timestamp", "transactTime", "tradeTime"):
            if key not in trade:
                continue
            ts = RealtimeRunner._timestamp_to_epoch(trade.get(key))
            if ts is not None:
                return int(ts * 1000.0)
        return None

    @staticmethod
    def _is_entry_trade_for_side(trade: Dict[str, Any], side: str) -> bool:
        position_side = str(trade.get("positionSide", "") or "").upper()
        side_value = str(trade.get("side", "") or "").upper()
        if side == "long":
            if position_side == "LONG":
                return side_value == "BUY"
            if position_side in ("BOTH", ""):
                return side_value == "BUY"
        if side == "short":
            if position_side == "SHORT":
                return side_value == "SELL"
            if position_side in ("BOTH", ""):
                return side_value == "SELL"
        return False

    def _apply_entry_trade_cache(
        self, side: str, raw_position: Dict[str, Any], trade: Dict[str, Any]
    ) -> None:
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        cache = dict(self._entry_cache.get(side) or {})
        entry_price = _as_float(raw_position.get("entry_price"))
        if entry_price is not None and entry_price > 0 and "entry_price" not in cache:
            cache["entry_price"] = entry_price
        qty = _as_float(raw_position.get("qty"))
        if qty is not None and qty > 0:
            cache.setdefault("qty", qty)
        direction = raw_position.get("direction")
        if isinstance(direction, (int, float)) and int(direction) != 0:
            cache.setdefault("direction", int(direction))
        leverage = _as_float(raw_position.get("leverage"))
        if leverage is not None:
            cache.setdefault("leverage", leverage)
        trade_id = trade.get("id") or trade.get("tradeId")
        if trade_id not in (None, ""):
            cache["entry_trade_id"] = trade_id
        order_id = trade.get("orderId") or trade.get("order_id")
        if order_id not in (None, ""):
            cache["entry_order_id"] = order_id
        trade_time_ms = self._trade_time_ms(trade)
        entry_time = self._timestamp_to_iso(trade_time_ms) if trade_time_ms else None
        if entry_time and cache.get("entry_time") in (None, ""):
            cache["entry_time"] = entry_time
        if entry_time and cache.get("entry_reason") in (None, "", "exchange_sync"):
            cache["entry_time"] = entry_time
        if cache.get("entry_reason") in (None, "", "exchange_sync"):
            cache["entry_reason"] = "exchange_fill"
        cache["updated_at"] = time.time()
        if cache:
            self._entry_cache[side] = cache

    def _update_entry_cache_from_trade(self, trade: Dict[str, Any]) -> None:
        trade_type = str(trade.get("type", ""))
        if trade_type not in {"entry", "exit", "exit_partial"}:
            return
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            return
        side = "long" if direction > 0 else "short"
        if trade_type == "exit":
            self._entry_cache[side] = {}
            return
        try:
            entry_price = float(trade.get("price", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        if entry_price <= 0:
            return
        cached: Dict[str, Any] = {"entry_price": entry_price, "updated_at": time.time()}
        try:
            leverage = float(trade.get("leverage"))
        except (TypeError, ValueError):
            leverage = None
        if leverage is not None and math.isfinite(leverage):
            cached["leverage"] = leverage
        try:
            qty = float(trade.get("qty"))
        except (TypeError, ValueError):
            qty = None
        if qty is not None and math.isfinite(qty):
            cached["qty"] = qty
        cached["direction"] = direction
        entry_time = trade.get("timestamp") or trade.get("entry_time")
        if entry_time not in (None, ""):
            cached["entry_time"] = entry_time
        entry_reason = trade.get("entry_reason") or trade.get("reason")
        if entry_reason not in (None, ""):
            cached["entry_reason"] = str(entry_reason)
        if len(cached) > 1:
            self._entry_cache[side] = cached

    def _update_entry_cache_from_order(self, trade: Dict[str, Any], order_id: int) -> None:
        trade_type = str(trade.get("type", ""))
        if trade_type != "entry":
            return
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            return
        side = "long" if direction > 0 else "short"
        cache = dict(self._entry_cache.get(side) or {})
        cache["entry_order_id"] = order_id
        entry_time = trade.get("timestamp") or trade.get("entry_time")
        if entry_time not in (None, ""):
            cache.setdefault("entry_time", entry_time)
        if cache:
            cache["updated_at"] = time.time()
            self._entry_cache[side] = cache

    def _apply_entry_cache(
        self, exchange_positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(exchange_positions, dict):
            return exchange_positions

        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        def _maybe_set(target: Dict[str, Any], key: str, source: Dict[str, Any]) -> None:
            if target.get(key) not in (None, "", 0, 0.0):
                return
            value = source.get(key)
            if value not in (None, "", 0, 0.0):
                target[key] = value

        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                self._entry_cache[side] = {}
                continue
            qty_val = _as_float(raw.get("qty", 0.0)) or 0.0
            if qty_val <= 0:
                self._entry_cache[side] = {}
                continue
            cache = self._entry_cache.get(side) or {}
            if not cache:
                continue
            ex_entry = _as_float(raw.get("entry_price"))
            cache_entry = _as_float(cache.get("entry_price"))
            if ex_entry and cache_entry:
                diff = abs(ex_entry - cache_entry) / ex_entry
                if diff > 0.01:
                    continue
            _maybe_set(raw, "entry_time", cache)
            _maybe_set(raw, "entry_reason", cache)
            _maybe_set(raw, "leverage", cache)
            _maybe_set(raw, "direction", cache)
            _maybe_set(raw, "qty", cache)
            _maybe_set(raw, "target_qty", cache)
            _maybe_set(raw, "part_qty", cache)
            _maybe_set(raw, "base_parts", cache)
            _maybe_set(raw, "parts_filled", cache)
            _maybe_set(raw, "entry_trade_id", cache)
            _maybe_set(raw, "entry_order_id", cache)
            exchange_positions[side] = raw
        return exchange_positions

    def _seed_exchange_exit_trades(
        self,
        result: Any,
        data: pd.DataFrame,
        exchange_positions: Dict[str, Dict[str, Any]],
        existing_trades: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if data.empty or not isinstance(exchange_positions, dict):
            return []
        last_row = data.iloc[-1]
        try:
            high_price = float(last_row.get("high", last_row.get("close", 0.0)) or 0.0)
            low_price = float(last_row.get("low", last_row.get("close", 0.0)) or 0.0)
            close_price = float(last_row.get("close", 0.0) or 0.0)
        except (TypeError, ValueError):
            return []
        timestamp = last_row.get("timestamp", time.time())
        open_positions = result.open_positions or {}
        existing_exit_sides: set[str] = set()
        emit_after_ts = self._emit_after_ts
        for trade in existing_trades or []:
            if str(trade.get("type")) not in {"exit", "exit_partial"}:
                continue
            if emit_after_ts is not None:
                ts_val = self._trade_timestamp_seconds(trade)
                if ts_val is not None and ts_val <= emit_after_ts:
                    continue
            direction = int(trade.get("direction", 0) or 0)
            if direction == 0:
                continue
            existing_exit_sides.add("long" if direction > 0 else "short")
        seeded: List[Dict[str, Any]] = []
        for side in ("long", "short"):
            ex = exchange_positions.get(side, {})
            if not isinstance(ex, dict) or not ex:
                continue
            entry_reason = str(ex.get("entry_reason") or "").lower()
            if entry_reason in {"manual_test", "manual"} and not ex.get("manual_test_targets"):
                continue
            try:
                qty = float(ex.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue
            if side in existing_exit_sides:
                if not (entry_reason in {"manual_test", "manual"} and ex.get("manual_test_targets")):
                    continue
            eng = open_positions.get(side, {})
            if isinstance(eng, dict):
                try:
                    eng_qty = float(eng.get("qty", 0.0) or 0.0)
                except (TypeError, ValueError):
                    eng_qty = 0.0
                if eng_qty > 0:
                    continue
            direction = int(ex.get("direction", 0) or 0)
            if direction == 0:
                direction = 1 if side == "long" else -1
            tp_price = ex.get("rr_tp_price")
            if tp_price is None:
                tp_price = ex.get("tp_price")
            sl_price = ex.get("rr_stop_price")
            if sl_price is None:
                sl_price = ex.get("stop_price")
            if sl_price is None:
                sl_price = ex.get("sl_price")
            try:
                tp_val = float(tp_price) if tp_price not in (None, "") else None
            except (TypeError, ValueError):
                tp_val = None
            try:
                sl_val = float(sl_price) if sl_price not in (None, "") else None
            except (TypeError, ValueError):
                sl_val = None
            exit_reason = None
            exit_price = None
            if sl_val is not None and sl_val > 0:
                if direction > 0 and low_price <= sl_val:
                    exit_reason = "seeded_stop"
                    exit_price = sl_val
                elif direction < 0 and high_price >= sl_val:
                    exit_reason = "seeded_stop"
                    exit_price = sl_val
            if exit_reason is None and tp_val is not None and tp_val > 0:
                if direction > 0 and high_price >= tp_val:
                    exit_reason = "seeded_tp"
                    exit_price = tp_val
                elif direction < 0 and low_price <= tp_val:
                    exit_reason = "seeded_tp"
                    exit_price = tp_val
            if exit_reason is None and (tp_val is None or tp_val <= 0) and (sl_val is None or sl_val <= 0):
                exit_reason = "seeded_no_targets"
                exit_price = close_price
            if exit_reason is None or exit_price is None:
                continue
            if exit_reason == "seeded_no_targets" and self.group == "live":
                continue
            try:
                entry_price = float(ex.get("entry_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            try:
                leverage = float(ex.get("leverage", self.leverage) or self.leverage or 1.0)
            except (TypeError, ValueError):
                leverage = self.leverage or 1.0
            leverage = leverage if leverage > 0 else 1.0
            pnl = 0.0
            ret = 0.0
            min_return = ex.get("min_return")
            max_return = ex.get("max_return")
            if entry_price > 0 and exit_price > 0:
                pnl = (exit_price - entry_price) * qty * direction
                notional = abs(entry_price * qty)
                ret = (pnl / notional) * leverage if notional > 0 else 0.0
                if min_return is None or max_return is None:
                    ret_high = ((high_price - entry_price) / entry_price) * direction * leverage
                    ret_low = ((low_price - entry_price) / entry_price) * direction * leverage
                    computed_min = min(ret_high, ret_low)
                    computed_max = max(ret_high, ret_low)
                    if min_return is None:
                        min_return = computed_min
                    if max_return is None:
                        max_return = computed_max
            seeded.append(
                {
                    "type": "exit",
                    "direction": direction,
                    "entry_price": entry_price,
                    "price": exit_price,
                    "qty": qty,
                    "fee": 0.0,
                    "leverage": leverage,
                    "pnl": pnl,
                    "return": ret,
                    "min_return": min_return,
                    "max_return": max_return,
                    "entry_time": ex.get("entry_time"),
                    "entry_reason": ex.get("entry_reason", "seeded"),
                    "exit_reason": exit_reason,
                    "timestamp": timestamp,
                    "stop_price": ex.get("stop_price") or ex.get("sl_price"),
                    "rr_stop_price": ex.get("rr_stop_price"),
                    "rr_tp_price": ex.get("rr_tp_price"),
                    "positionSide": "LONG" if direction > 0 else "SHORT",
                }
            )
        return seeded

    def _seed_exchange_flat_exits(
        self,
        result: Any,
        data: pd.DataFrame,
        exchange_positions: Dict[str, Dict[str, Any]],
        existing_trades: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if data.empty or not isinstance(exchange_positions, dict):
            return []
        if result is None:
            return []
        last_row = data.iloc[-1]
        try:
            high_price = float(last_row.get("high", last_row.get("close", 0.0)) or 0.0)
            low_price = float(last_row.get("low", last_row.get("close", 0.0)) or 0.0)
            close_price = float(last_row.get("close", 0.0) or 0.0)
        except (TypeError, ValueError):
            return []
        timestamp = last_row.get("timestamp", time.time())
        open_positions = result.open_positions or {}
        existing_exit_sides: set[str] = set()
        emit_after_ts = self._emit_after_ts
        for trade in existing_trades or []:
            if str(trade.get("type")) not in {"exit", "exit_partial"}:
                continue
            if emit_after_ts is not None:
                ts_val = self._trade_timestamp_seconds(trade)
                if ts_val is not None and ts_val <= emit_after_ts:
                    continue
            direction = int(trade.get("direction", 0) or 0)
            if direction == 0:
                continue
            existing_exit_sides.add("long" if direction > 0 else "short")
        seeded: List[Dict[str, Any]] = []
        for side in ("long", "short"):
            eng = open_positions.get(side, {})
            if not isinstance(eng, dict) or not eng:
                continue
            entry_reason = str(eng.get("entry_reason", "") or "").lower()
            if self.group == "live" and (
                entry_reason.startswith("cheatkey_") or "expected" in entry_reason
            ):
                # Skip synthetic/expected engine positions to avoid duplicate live result logs.
                continue
            try:
                eng_qty = float(eng.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                eng_qty = 0.0
            if eng_qty <= 0:
                continue
            if side in existing_exit_sides:
                continue
            ex = exchange_positions.get(side, {})
            try:
                ex_qty = float(ex.get("qty", 0.0) or 0.0) if isinstance(ex, dict) else 0.0
            except (TypeError, ValueError):
                ex_qty = 0.0
            if ex_qty > 0:
                continue
            direction = int(eng.get("direction", 0) or 0)
            if direction == 0:
                direction = 1 if side == "long" else -1
            try:
                entry_price = float(eng.get("entry_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            try:
                leverage = float(eng.get("leverage", self.leverage) or self.leverage or 1.0)
            except (TypeError, ValueError):
                leverage = self.leverage or 1.0
            leverage = leverage if leverage > 0 else 1.0
            pnl = 0.0
            ret = 0.0
            min_return = None
            max_return = None
            if entry_price > 0 and close_price > 0:
                pnl = (close_price - entry_price) * eng_qty * direction
                notional = abs(entry_price * eng_qty)
                ret = (pnl / notional) * leverage if notional > 0 else 0.0
                ret_high = ((high_price - entry_price) / entry_price) * direction * leverage
                ret_low = ((low_price - entry_price) / entry_price) * direction * leverage
                min_return = min(ret_high, ret_low)
                max_return = max(ret_high, ret_low)
            seeded.append(
                {
                    "type": "exit",
                    "direction": direction,
                    "entry_price": entry_price,
                    "price": close_price,
                    "qty": eng_qty,
                    "fee": 0.0,
                    "leverage": leverage,
                    "pnl": pnl,
                    "return": ret,
                    "min_return": min_return,
                    "max_return": max_return,
                    "entry_time": eng.get("entry_time"),
                    "entry_reason": eng.get("entry_reason", "engine_open"),
                    "exit_reason": "exchange_closed",
                    "timestamp": timestamp,
                    "positionSide": "LONG" if direction > 0 else "SHORT",
                }
            )
        return seeded

    def _seed_exchange_signal_exits(
        self,
        data: pd.DataFrame,
        exchange_positions: Dict[str, Dict[str, Any]],
        existing_trades: List[Dict[str, Any]],
        exit_signals: pd.Series,
        slope_exit_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if data.empty or not isinstance(exchange_positions, dict):
            return []
        if exit_signals is None or len(exit_signals) == 0:
            return []
        try:
            exit_signal = int(exit_signals.iloc[-1])
        except (TypeError, ValueError):
            exit_signal = 0
        if exit_signal == 0:
            return []
        last_row = data.iloc[-1]
        timestamp = last_row.get("timestamp", time.time())
        try:
            close_price = float(last_row.get("close", 0.0) or 0.0)
        except (TypeError, ValueError):
            close_price = 0.0
        if close_price <= 0:
            return []
        try:
            high_price = float(last_row.get("high", 0.0) or 0.0)
        except (TypeError, ValueError):
            high_price = 0.0
        try:
            low_price = float(last_row.get("low", 0.0) or 0.0)
        except (TypeError, ValueError):
            low_price = 0.0

        existing_exit_sides: set[str] = set()
        emit_after_ts = self._emit_after_ts
        for trade in existing_trades or []:
            if str(trade.get("type")) not in {"exit", "exit_partial"}:
                continue
            if emit_after_ts is not None:
                ts_val = self._trade_timestamp_seconds(trade)
                if ts_val is not None and ts_val <= emit_after_ts:
                    continue
            direction = int(trade.get("direction", 0) or 0)
            if direction == 0:
                continue
            existing_exit_sides.add("long" if direction > 0 else "short")

        target_side = "long" if exit_signal > 0 else "short"
        if target_side in existing_exit_sides:
            return []
        raw = exchange_positions.get(target_side, {})
        if not isinstance(raw, dict) or not raw:
            return []
        raw_direction = int(raw.get("direction", 0) or 0)
        if raw_direction != 0 and raw_direction != exit_signal:
            return []
        entry_reason = str(raw.get("entry_reason") or "").lower()
        if entry_reason in {"manual_test", "manual"}:
            return []
        try:
            qty = float(raw.get("qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            return []
        try:
            entry_price = float(raw.get("entry_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        try:
            leverage = float(raw.get("leverage", self.leverage) or self.leverage or 1.0)
        except (TypeError, ValueError):
            leverage = self.leverage or 1.0
        leverage = leverage if leverage > 0 else 1.0
        direction = 1 if target_side == "long" else -1
        parts_filled = raw.get("parts_filled")
        if not isinstance(parts_filled, (int, float)):
            entry_cache = self._entry_cache.get(target_side) or {}
            parts_filled = entry_cache.get("parts_filled")
        try:
            parts_filled_val = int(parts_filled) if parts_filled is not None else None
        except (TypeError, ValueError):
            parts_filled_val = None

        slope_exit_mode = False
        slope_exit_cooldown_bars = None
        slope_exit_pnl_threshold = None
        slope_exit_use_threshold = True
        slope_exit_ignore_threshold_after_add = False
        if isinstance(slope_exit_config, dict):
            slope_exit_mode = bool(slope_exit_config.get("mode"))
            slope_exit_cooldown_bars = slope_exit_config.get("cooldown_bars")
            slope_exit_pnl_threshold = slope_exit_config.get("pnl_threshold")
            slope_exit_use_threshold = bool(slope_exit_config.get("use_threshold", True))
            slope_exit_ignore_threshold_after_add = bool(
                slope_exit_config.get("ignore_threshold_after_add", False)
            )

        can_exit = True
        reason = "mode_disabled"
        cooldown_bars_val = None
        bars_since_entry = None
        threshold_target_price = None
        if slope_exit_mode:
            reason = "cooldown_ok"
            if slope_exit_cooldown_bars is not None:
                try:
                    cooldown_bars_val = int(slope_exit_cooldown_bars)
                except (TypeError, ValueError):
                    cooldown_bars_val = None
                if cooldown_bars_val is not None and cooldown_bars_val > 0:
                    entry_ts = _normalize_timestamp_value(raw.get("entry_time"))
                    if entry_ts is None:
                        entry_cache = self._entry_cache.get(target_side) or {}
                        entry_ts = _normalize_timestamp_value(entry_cache.get("entry_time"))
                    last_ts = _normalize_timestamp_value(timestamp)
                    if entry_ts is None or last_ts is None:
                        can_exit = False
                        reason = "cooldown_unknown"
                    else:
                        interval_seconds = max(_interval_seconds(self.interval), 1)
                        bars_since_entry = int(max(0.0, last_ts - entry_ts) // interval_seconds)
                        if bars_since_entry < cooldown_bars_val:
                            can_exit = False
                            reason = "cooldown_active"
            if can_exit:
                if not slope_exit_use_threshold:
                    reason = "threshold_disabled"
                elif (
                    slope_exit_ignore_threshold_after_add
                    and parts_filled_val is not None
                    and parts_filled_val > 1
                ):
                    reason = "threshold_ignored_after_add"
                else:
                    if slope_exit_pnl_threshold is None:
                        reason = "threshold_unset"
                    else:
                        threshold_target_price = _target_price_for_pnl(
                            entry_price,
                            direction,
                            leverage,
                            float(slope_exit_pnl_threshold),
                        )
                        if threshold_target_price is None or high_price <= 0 or low_price <= 0:
                            can_exit = False
                            reason = "threshold_price_unavailable"
                        else:
                            threshold_ok = (
                                (direction > 0 and high_price >= threshold_target_price)
                                or (direction < 0 and low_price <= threshold_target_price)
                            )
                            can_exit = threshold_ok
                            reason = "threshold_met" if threshold_ok else "threshold_not_met"

        self._log_slope_exit_decision(
            side=target_side,
            timestamp=timestamp,
            can_exit=can_exit,
            reason=reason,
            slope_exit_mode=slope_exit_mode,
            exit_signal=exit_signal,
            entry_price=entry_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            leverage=leverage,
            pnl_threshold=slope_exit_pnl_threshold,
            target_price=threshold_target_price,
            cooldown_bars=cooldown_bars_val,
            bars_since_entry=bars_since_entry,
            parts_filled=parts_filled_val,
        )
        if not can_exit:
            return []
        pnl = 0.0
        ret = 0.0
        if entry_price > 0:
            pnl = (close_price - entry_price) * qty * direction
            notional = abs(entry_price * qty)
            ret = (pnl / notional) * leverage if notional > 0 else 0.0
        return [
            {
                "type": "exit",
                "direction": direction,
                "entry_price": entry_price,
                "price": close_price,
                "qty": qty,
                "fee": 0.0,
                "leverage": leverage,
                "pnl": pnl,
                "return": ret,
                "entry_time": raw.get("entry_time"),
                "entry_reason": raw.get("entry_reason", "exchange_sync"),
                "exit_reason": "signal_exit",
                "timestamp": timestamp,
                "stop_price": raw.get("stop_price") or raw.get("sl_price"),
                "rr_stop_price": raw.get("rr_stop_price"),
                "rr_tp_price": raw.get("rr_tp_price"),
                "positionSide": "LONG" if direction > 0 else "SHORT",
            }
        ]

    def _log_slope_exit_decision(
        self,
        *,
        side: str,
        timestamp: Any,
        can_exit: bool,
        reason: str,
        slope_exit_mode: bool,
        exit_signal: int,
        entry_price: float,
        close_price: float,
        high_price: float,
        low_price: float,
        leverage: float,
        pnl_threshold: Any,
        target_price: Optional[float],
        cooldown_bars: Optional[int],
        bars_since_entry: Optional[int],
        parts_filled: Optional[int],
    ) -> None:
        ts_val = _normalize_timestamp_value(timestamp)
        last_ts = self._slope_exit_log_ts.get(side)
        if ts_val is not None and last_ts == ts_val:
            return
        self._slope_exit_log_ts[side] = ts_val
        status = "allowed" if can_exit else "blocked"
        logger.info(
            "Live slope exit %s. symbol=%s side=%s signal=%s mode=%s reason=%s "
            "entry_price=%.8g close=%.8g high=%.8g low=%.8g leverage=%.4g "
            "pnl_threshold=%s target_price=%s cooldown_bars=%s bars_since_entry=%s parts_filled=%s",
            status,
            self.symbol,
            side,
            exit_signal,
            "on" if slope_exit_mode else "off",
            reason,
            entry_price,
            close_price,
            high_price,
            low_price,
            leverage,
            pnl_threshold,
            f"{target_price:.8g}" if target_price is not None else "n/a",
            cooldown_bars if cooldown_bars is not None else "n/a",
            bars_since_entry if bars_since_entry is not None else "n/a",
            parts_filled if parts_filled is not None else "n/a",
        )

    async def build_livetest_signal_exit_trade(
        self, direction: Optional[int], price: float = 0.0
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        data = self.get_chart_data(include_live=True)
        if data.empty:
            return None, "No chart data available."
        exchange_positions = await self._get_exchange_positions(force=True)
        if not isinstance(exchange_positions, dict):
            return None, "No position data returned."
        engine_positions: Dict[str, Dict[str, Any]] = {}
        if self.latest_payload and isinstance(self.latest_payload.positions, dict):
            engine_positions = self.latest_payload.positions
        if not (self.group == "live" and self.exchange_position_authority):
            exchange_positions = self._merge_exchange_positions(
                engine_positions, exchange_positions
            )
            exchange_positions = self._apply_target_cache(exchange_positions)
        exchange_positions = self._apply_entry_cache(exchange_positions)

        if direction not in (None, 1, -1):
            return None, "Usage: `!closetest [long|short] [price]`"

        def _qty_from_side(side_name: str) -> float:
            raw_side = exchange_positions.get(side_name, {})
            if not isinstance(raw_side, dict):
                return 0.0
            try:
                return float(raw_side.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        if direction is None:
            long_qty = _qty_from_side("long")
            short_qty = _qty_from_side("short")
            if long_qty <= 0 and short_qty <= 0:
                return None, "No matching open position to close."
            if long_qty > 0 and short_qty <= 0:
                direction = 1
            elif short_qty > 0 and long_qty <= 0:
                direction = -1
            else:
                return (
                    None,
                    "Usage: `!closetest [long|short] [price]` (multiple positions found)",
                )

        side = "long" if direction > 0 else "short"
        raw = exchange_positions.get(side, {}) if isinstance(exchange_positions, dict) else {}
        if not isinstance(raw, dict) or not raw:
            return None, "No matching open position to close."
        try:
            qty = float(raw.get("qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            return None, "No matching open position to close."

        if price <= 0:
            last_row = data.iloc[-1]
            try:
                price = float(last_row.get("close", 0.0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
        if price <= 0:
            return None, f"Price unavailable for {self.symbol}."

        try:
            entry_price = float(raw.get("entry_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        try:
            leverage = float(raw.get("leverage", self.leverage) or self.leverage or 1.0)
        except (TypeError, ValueError):
            leverage = self.leverage or 1.0
        leverage = leverage if leverage > 0 else 1.0
        pnl = 0.0
        ret = 0.0
        if entry_price > 0:
            pnl = (price - entry_price) * qty * direction
            notional = abs(entry_price * qty)
            ret = (pnl / notional) * leverage if notional > 0 else 0.0
        timestamp = data.iloc[-1].get("timestamp", time.time())
        trade = {
            "type": "exit",
            "direction": direction,
            "entry_price": entry_price,
            "price": price,
            "qty": qty,
            "fee": 0.0,
            "leverage": leverage,
            "pnl": pnl,
            "return": ret,
            "entry_time": raw.get("entry_time"),
            "entry_reason": raw.get("entry_reason", "exchange_sync"),
            "exit_reason": "signal_exit",
            "timestamp": timestamp,
            "stop_price": raw.get("stop_price") or raw.get("sl_price"),
            "rr_stop_price": raw.get("rr_stop_price"),
            "rr_tp_price": raw.get("rr_tp_price"),
            "positionSide": "LONG" if direction > 0 else "SHORT",
        }
        return trade, None

    async def set_livetest_targets(
        self,
        side: Optional[str],
        *,
        tp_pct: float,
        sl_pct: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        if tp_pct <= 0 or sl_pct <= 0:
            return None, "TP/SL 수익률은 0보다 큰 값이어야 합니다."

        if side not in (None, "long", "short"):
            return None, "Usage: `!livetest settpsl [long|short]`"

        exchange_positions = await self._get_exchange_positions(force=True)
        if not isinstance(exchange_positions, dict):
            exchange_positions = {}
        exchange_positions = self._apply_entry_cache(exchange_positions)

        def _position_info(target_side: str) -> Dict[str, Any]:
            entry = self._entry_cache.get(target_side) or {}
            raw = exchange_positions.get(target_side, {}) if exchange_positions else {}
            info = {
                "entry_price": _as_float(entry.get("entry_price")),
                "leverage": _as_float(entry.get("leverage")),
                "qty": _as_float(entry.get("qty")),
                "entry_reason": str(entry.get("entry_reason") or ""),
            }
            if info["entry_price"] in (None, 0.0):
                info["entry_price"] = _as_float(raw.get("entry_price"))
            if info["leverage"] in (None, 0.0):
                info["leverage"] = _as_float(raw.get("leverage"))
            if info["qty"] in (None, 0.0):
                info["qty"] = _as_float(raw.get("qty"))
            if not info["entry_reason"]:
                info["entry_reason"] = str(raw.get("entry_reason") or "")
            if info["leverage"] in (None, 0.0):
                info["leverage"] = self._resolve_effective_leverage()
            return info

        if side is None:
            candidates = []
            for candidate in ("long", "short"):
                info = _position_info(candidate)
                qty_val = info.get("qty") or 0.0
                if qty_val > 0:
                    candidates.append(candidate)
            if not candidates:
                return None, "TP/SL을 설정할 포지션이 없습니다."
            if len(candidates) > 1:
                return None, "롱/숏 포지션이 둘 다 열려있습니다. `long` 또는 `short`를 지정해주세요."
            side = candidates[0]

        info = _position_info(side)
        qty_val = info.get("qty") or 0.0
        if qty_val <= 0:
            return None, "TP/SL을 설정할 포지션이 없습니다."

        entry_price = info.get("entry_price") or 0.0
        if entry_price <= 0:
            return None, "진입가를 확인할 수 없습니다."

        leverage = info.get("leverage") or 1.0
        if leverage <= 0:
            leverage = 1.0

        direction = 1 if side == "long" else -1
        tp_unlevered = (tp_pct / 100.0) / leverage
        sl_unlevered = (sl_pct / 100.0) / leverage
        tp_price = entry_price * (1.0 + tp_unlevered * direction)
        sl_price = entry_price * (1.0 - sl_unlevered * direction)

        cache = dict(self._target_cache.get(side) or {})
        cache["entry_price"] = entry_price
        cache["tp_price"] = tp_price
        cache["sl_price"] = sl_price
        cache["rr_tp_price"] = tp_price
        cache["rr_stop_price"] = sl_price
        cache["manual_test_targets"] = True
        cache["updated_at"] = time.time()
        self._target_cache[side] = cache

        if isinstance(self._exchange_positions, dict) and self._exchange_positions:
            self._exchange_positions = self._apply_target_cache(self._exchange_positions)
        if self.latest_payload is not None and isinstance(self.latest_payload.positions, dict):
            positions = self._apply_target_cache(copy.deepcopy(self.latest_payload.positions))
            payload = RunnerPayload(
                symbol=self.latest_payload.symbol,
                interval=self.latest_payload.interval,
                price=self.latest_payload.price,
                equity=self.latest_payload.equity,
                free_balance=self.latest_payload.free_balance,
                positions=positions,
                engine_positions=self.latest_payload.engine_positions,
                updated_at=time.time(),
            )
            self.latest_payload = payload
            if self.on_payload:
                self.on_payload(self.group, payload)
        if self.state_path:
            await asyncio.to_thread(self._persist_state)

        return (
            {
                "side": side,
                "entry_price": entry_price,
                "leverage": leverage,
                "tp_price": tp_price,
                "sl_price": sl_price,
            },
            None,
        )

    @staticmethod
    def _merge_exchange_positions(
        engine_positions: Dict[str, Dict[str, Any]],
        exchange_positions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(exchange_positions, dict):
            return exchange_positions
        if not isinstance(engine_positions, dict):
            return exchange_positions
        merged: Dict[str, Dict[str, Any]] = {
            "long": dict(exchange_positions.get("long") or {}),
            "short": dict(exchange_positions.get("short") or {}),
        }

        def _is_missing(value: Any) -> bool:
            return value in (None, "", 0, 0.0)

        def _maybe_set(target: Dict[str, Any], key: str, source: Dict[str, Any], *alts: str) -> None:
            if not _is_missing(target.get(key)):
                return
            for name in (key, *alts):
                value = source.get(name)
                if not _is_missing(value):
                    target[key] = value
                    return

        for side in ("long", "short"):
            ex = merged.get(side, {})
            if not ex:
                continue
            eng = engine_positions.get(side, {})
            if not isinstance(eng, dict) or not eng:
                continue

            _maybe_set(ex, "rr_tp_price", eng, "tp_price")
            _maybe_set(ex, "tp_price", eng, "rr_tp_price")
            _maybe_set(ex, "rr_stop_price", eng, "stop_price", "sl_price")
            _maybe_set(ex, "stop_price", eng, "rr_stop_price", "sl_price")
            _maybe_set(ex, "sl_price", eng, "rr_stop_price", "stop_price")
            _maybe_set(ex, "entry_time", eng)
            _maybe_set(ex, "entry_reason", eng)
            _maybe_set(ex, "min_return", eng)
            _maybe_set(ex, "max_return", eng)
            _maybe_set(ex, "prev_extreme_stop_price", eng)
            merged[side] = ex

        return merged

    async def _refresh_exchange_balance(self) -> Optional[Dict[str, float]]:
        if self.binance is None:
            return self._exchange_balance
        try:
            raw_account = await asyncio.to_thread(self.binance.get_account)
            parsed_account = self._parse_exchange_account(raw_account)
            if parsed_account:
                self._exchange_balance = parsed_account
            else:
                logger.warning("Exchange account payload parsed empty.")
        except Exception as exc:
            details = _format_http_error(exc)
            logger.warning("Exchange account sync failed: %s.%s", exc, details)
        return self._exchange_balance

    async def _resolve_engine_initial_balance(self) -> float:
        if self.group != "live" or self.binance is None:
            return self.initial_balance
        if self._exchange_balance is None or self._should_sync_exchange():
            await self._refresh_exchange_balance()
        available = None
        equity = None
        if self._exchange_balance:
            available = self._exchange_balance.get("available_balance")
            equity = self._exchange_balance.get("equity")
        candidate = None
        if isinstance(equity, (int, float)) and equity > 0:
            candidate = float(equity)
        elif isinstance(available, (int, float)) and available > 0:
            candidate = float(available)

        if self._has_open_positions():
            if self._sizing_equity is None and candidate is not None:
                self._sizing_equity = candidate
            return self._sizing_equity if self._sizing_equity is not None else self.initial_balance

        if candidate is not None:
            self._sizing_equity = candidate
            return candidate
        return self.initial_balance

    def _has_open_positions(self, position_side: Optional[str] = None) -> bool:
        positions: Optional[Dict[str, Dict[str, Any]]] = None
        if self.sync_exchange_positions and self._exchange_positions is not None:
            positions = self._exchange_positions
        elif self.latest_payload is not None:
            positions = self.latest_payload.positions
        if not isinstance(positions, dict):
            return False

        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        normalized_side = self._normalize_position_side(position_side)
        if normalized_side == "LONG":
            sides = ("long",)
        elif normalized_side == "SHORT":
            sides = ("short",)
        else:
            sides = ("long", "short")

        for side in sides:
            raw = positions.get(side)
            if not isinstance(raw, dict) or not raw:
                continue
            if raw.get("active") is True:
                return True
            if _as_float(raw.get("qty", 0.0)) > 0:
                return True
        return False

    def _parse_exchange_positions(
        self, raw_positions: Any
    ) -> Dict[str, Dict[str, Any]]:
        positions: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        min_notional = (
            self._symbol_filters.min_notional
            if self._symbol_filters and self._symbol_filters.min_notional
            else None
        )
        step_size = (
            self._symbol_filters.step_size
            if self._symbol_filters and self._symbol_filters.step_size
            else None
        )
        if isinstance(raw_positions, dict):
            raw_positions = [raw_positions]
        if not isinstance(raw_positions, list):
            return positions
        for item in raw_positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "") or "").upper()
            if symbol and symbol != self.symbol:
                continue
            try:
                pos_amt = float(item.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos_amt = 0.0
            if pos_amt == 0:
                continue
            position_side = str(item.get("positionSide", "BOTH") or "BOTH").upper()
            side = None
            direction = 0
            if position_side == "LONG":
                side = "long"
                direction = 1
            elif position_side == "SHORT":
                side = "short"
                direction = -1
            else:
                if pos_amt > 0:
                    side = "long"
                    direction = 1
                else:
                    side = "short"
                    direction = -1
            if side is None or direction == 0:
                continue
            try:
                entry_price = float(item.get("entryPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            try:
                mark_price = float(item.get("markPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                mark_price = 0.0
            try:
                leverage = float(item.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            qty = abs(pos_amt)
            if step_size is not None and step_size > 0 and qty < step_size:
                continue
            reference_price = entry_price if entry_price > 0 else mark_price
            if min_notional is not None and reference_price > 0:
                notional = abs(reference_price * qty)
                if notional < min_notional:
                    continue
            exchange_update_time = item.get("updateTime") or item.get("update_time")
            positions[side] = {
                "entry_price": entry_price,
                "qty": qty,
                "direction": direction,
                "leverage": leverage,
                "entry_time": None,
                "exchange_update_time": exchange_update_time,
                "exchange": True,
                "positionSide": position_side if position_side in {"LONG", "SHORT"} else None,
            }
        return positions

    def _extract_dust_positions(
        self, raw_positions: Any
    ) -> Dict[str, Dict[str, Any]]:
        dust: Dict[str, Dict[str, Any]] = {}
        if self._symbol_filters is None:
            return dust
        min_notional = self._symbol_filters.min_notional
        step_size = self._symbol_filters.step_size
        if min_notional is None and (step_size is None or step_size <= 0):
            return dust
        if isinstance(raw_positions, dict):
            raw_positions = [raw_positions]
        if not isinstance(raw_positions, list):
            return dust
        for item in raw_positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "") or "").upper()
            if symbol and symbol != self.symbol:
                continue
            try:
                pos_amt = float(item.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos_amt = 0.0
            if pos_amt == 0:
                continue
            position_side = str(item.get("positionSide", "BOTH") or "BOTH").upper()
            side = None
            direction = 0
            if position_side == "LONG":
                side = "long"
                direction = 1
            elif position_side == "SHORT":
                side = "short"
                direction = -1
            else:
                if pos_amt > 0:
                    side = "long"
                    direction = 1
                else:
                    side = "short"
                    direction = -1
            if side is None or direction == 0:
                continue
            try:
                entry_price = float(item.get("entryPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            try:
                mark_price = float(item.get("markPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                mark_price = 0.0
            qty = abs(pos_amt)
            reference_price = entry_price if entry_price > 0 else mark_price
            below_step = step_size is not None and step_size > 0 and qty < step_size
            below_notional = (
                min_notional is not None
                and reference_price > 0
                and abs(reference_price * qty) < min_notional
            )
            if not (below_step or below_notional):
                continue
            dust[side] = {
                "qty": qty,
                "direction": direction,
                "positionSide": position_side if position_side in {"LONG", "SHORT"} else None,
            }
        return dust

    async def _maybe_submit_dust_closes(
        self,
        dust_positions: Dict[str, Dict[str, Any]],
        engine_positions: Optional[Dict[str, Dict[str, Any]]],
    ) -> None:
        if not dust_positions:
            return
        if self.group != "live" or not self.auto_trade:
            return
        if self.binance is None:
            return
        tick_size = self._symbol_filters.tick_size if self._symbol_filters else 0.0
        now = time.time()
        for side in ("long", "short"):
            raw = dust_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                continue
            if side in self._pending_exchange_exits:
                continue
            eng_qty = 0.0
            if isinstance(engine_positions, dict):
                eng_raw = engine_positions.get(side, {})
                try:
                    eng_qty = (
                        float(eng_raw.get("qty", 0.0) or 0.0)
                        if isinstance(eng_raw, dict)
                        else 0.0
                    )
                except (TypeError, ValueError):
                    eng_qty = 0.0
            if eng_qty > 0:
                continue
            last_ts = self._last_dust_close_ts.get(side)
            if last_ts is not None and (now - last_ts) < 60.0:
                continue
            order_side = "SELL" if side == "long" else "BUY"
            position_side = raw.get("positionSide")
            result = await self._submit_dust_close(order_side, position_side, tick_size)
            self._last_dust_close_ts[side] = now
            if result.success:
                logger.warning(
                    "Dust close submitted after exit sync. symbol=%s side=%s qty=%.8g positionSide=%s",
                    self.symbol,
                    side,
                    float(raw.get("qty", 0.0) or 0.0),
                    position_side,
                )
            else:
                logger.warning(
                    "Dust close failed after exit sync. symbol=%s side=%s detail=%s",
                    self.symbol,
                    side,
                    result.detail,
                )

    def _record_position_sync(
        self,
        engine_positions: Dict[str, Dict[str, Any]],
        exchange_positions: Dict[str, Dict[str, Any]],
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "group": self.group,
            "symbol": self.symbol,
            "engine_positions": self._serialize_positions(engine_positions),
            "exchange_positions": self._serialize_positions(exchange_positions),
        }
        snapshot = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        if snapshot == self._last_sync_snapshot:
            return
        self._last_sync_snapshot = snapshot
        if not self.sync_history_path:
            return
        try:
            path = Path(self.sync_history_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(snapshot)
                handle.write("\n")
        except Exception:
            return

    @staticmethod
    def _parse_exchange_account(raw_account: Any) -> Dict[str, float]:
        if not isinstance(raw_account, dict):
            return {}
        def _as_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        available = _as_float(raw_account.get("availableBalance"))
        wallet = _as_float(raw_account.get("totalWalletBalance"))
        unrealized = _as_float(raw_account.get("totalUnrealizedProfit"))
        margin = _as_float(raw_account.get("totalMarginBalance"))
        equity = None
        if margin is not None and margin > 0:
            equity = margin
        elif wallet is not None or unrealized is not None:
            equity = (wallet or 0.0) + (unrealized or 0.0)
        parsed: Dict[str, float] = {}
        if available is not None:
            parsed["available_balance"] = available
        if equity is not None:
            parsed["equity"] = equity
        return parsed

    @staticmethod
    def _extract_symbol_risk(raw_positions: Any, symbol: str) -> Dict[str, Any]:
        if isinstance(raw_positions, list):
            for item in raw_positions:
                if not isinstance(item, dict):
                    continue
                if str(item.get("symbol", "") or "").upper() == symbol.upper():
                    return item
            return {}
        if isinstance(raw_positions, dict):
            return raw_positions
        return {}

    @staticmethod
    def _exchange_has_open_position(raw_positions: Any, symbol: str) -> bool:
        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        symbol = symbol.upper()
        if isinstance(raw_positions, list):
            for item in raw_positions:
                if not isinstance(item, dict):
                    continue
                item_symbol = str(item.get("symbol", "") or "").upper()
                if item_symbol and item_symbol != symbol:
                    continue
                if abs(_as_float(item.get("positionAmt", 0.0))) > 0:
                    return True
            return False
        if isinstance(raw_positions, dict):
            item_symbol = str(raw_positions.get("symbol", "") or "").upper()
            if item_symbol and item_symbol != symbol:
                return False
            return abs(_as_float(raw_positions.get("positionAmt", 0.0))) > 0
        return False

    def _exchange_has_open_position_side(
        self, raw_positions: Any, symbol: str, position_side: Optional[str]
    ) -> bool:
        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        normalized_side = self._normalize_position_side(position_side)
        if normalized_side not in ("LONG", "SHORT"):
            return self._exchange_has_open_position(raw_positions, symbol)

        symbol = symbol.upper()
        if isinstance(raw_positions, list):
            for item in raw_positions:
                if not isinstance(item, dict):
                    continue
                item_symbol = str(item.get("symbol", "") or "").upper()
                if item_symbol and item_symbol != symbol:
                    continue
                amt = _as_float(item.get("positionAmt", 0.0))
                if normalized_side == "LONG" and amt > 0:
                    return True
                if normalized_side == "SHORT" and amt < 0:
                    return True
            return False
        if isinstance(raw_positions, dict):
            item_symbol = str(raw_positions.get("symbol", "") or "").upper()
            if item_symbol and item_symbol != symbol:
                return False
            amt = _as_float(raw_positions.get("positionAmt", 0.0))
            if normalized_side == "LONG":
                return amt > 0
            return amt < 0
        return False

    async def _ensure_live_leverage(self, desired_leverage: float) -> Optional[float]:
        if self.group != "live" or self.binance is None:
            return None
        desired_int = int(round(desired_leverage)) if desired_leverage > 0 else 1
        try:
            payload = await asyncio.to_thread(self.binance.get_position_risk, self.symbol)
        except Exception as exc:
            logger.warning(
                "Live leverage check failed: symbol=%s desired=%s error=%s",
                self.symbol,
                desired_int,
                _format_binance_error(exc) or str(exc),
            )
            return None
        risk = self._extract_symbol_risk(payload, self.symbol)
        try:
            current = float(risk.get("leverage", 0.0) or 0.0)
        except (TypeError, ValueError):
            current = 0.0
        if current <= 0:
            return None
        if int(round(current)) != desired_int:
            margin_type = risk.get("marginType")
            max_notional = risk.get("maxNotionalValue")
            try:
                await asyncio.to_thread(self.binance.set_leverage, self.symbol, desired_int)
            except Exception as exc:
                logger.warning(
                    "Live leverage set failed: symbol=%s desired=%s current=%s "
                    "margin_type=%s max_notional=%s error=%s",
                    self.symbol,
                    desired_int,
                    current,
                    margin_type,
                    "n/a" if max_notional is None else str(max_notional),
                    _format_binance_error(exc) or str(exc),
                )
                return float(current)
            try:
                refreshed = await asyncio.to_thread(
                    self.binance.get_position_risk, self.symbol
                )
            except Exception:
                return float(current)
            updated = self._extract_symbol_risk(refreshed, self.symbol)
            try:
                after = float(updated.get("leverage", 0.0) or 0.0)
            except (TypeError, ValueError):
                after = 0.0
            if after > 0 and int(round(after)) != desired_int:
                logger.warning(
                    "Live leverage mismatch after set: symbol=%s desired=%s current=%s",
                    self.symbol,
                    desired_int,
                    after,
                )
                return float(after)
            return float(after) if after > 0 else float(current)
        return float(current)

    async def _log_live_margin_context(
        self,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool,
        error_detail: str,
    ) -> None:
        if self.group != "live" or self.binance is None:
            return
        details: List[str] = []
        try:
            payload = await asyncio.to_thread(self.binance.get_position_mode)
            if isinstance(payload, dict) and "dualSidePosition" in payload:
                details.append(f"dual_side={payload.get('dualSidePosition')}")
        except Exception as exc:
            details.append(f"position_mode_error={_format_binance_error(exc) or str(exc)}")
        try:
            payload = await asyncio.to_thread(self.binance.get_position_risk, self.symbol)
            risk = self._extract_symbol_risk(payload, self.symbol)
            if risk:
                if "leverage" in risk:
                    details.append(f"risk_leverage={risk.get('leverage')}")
                if "marginType" in risk:
                    details.append(f"risk_margin_type={risk.get('marginType')}")
                for key in (
                    "maxNotionalValue",
                    "positionAmt",
                    "initialMargin",
                    "maintMargin",
                    "isolatedMargin",
                    "notional",
                ):
                    if key in risk:
                        details.append(f"{key}={risk.get(key)}")
        except Exception as exc:
            details.append(f"position_risk_error={_format_binance_error(exc) or str(exc)}")
        try:
            payload = await asyncio.to_thread(self.binance.get_account)
            if isinstance(payload, dict):
                for key in (
                    "totalInitialMargin",
                    "totalPositionInitialMargin",
                    "totalOpenOrderInitialMargin",
                    "totalMarginBalance",
                    "availableBalance",
                    "totalWalletBalance",
                    "totalUnrealizedProfit",
                ):
                    if key in payload:
                        details.append(f"{key}={payload.get(key)}")
        except Exception as exc:
            details.append(f"account_error={_format_binance_error(exc) or str(exc)}")
        if details:
            logger.warning(
                "Live margin context: symbol=%s side=%s qty=%.8g price=%.8g reduce_only=%s "
                "error=%s %s",
                self.symbol,
                side,
                qty,
                price,
                reduce_only,
                error_detail,
                " ".join(details),
            )

    async def _log_live_exit_rejection_context(
        self,
        reason: str,
        error_detail: str,
        side: str,
        position_side: Optional[str],
        desired_position_side: Optional[str],
        direction: int,
        qty: float,
        remaining_qty: float,
        price: float,
        reduce_only: bool,
        use_post_only: bool,
        leverage: float,
        tick_size: float,
        step_size: float,
        attempt: int,
        exchange_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        refresh_idx: Optional[int] = None,
        retry_without_reduce_only: bool = False,
    ) -> None:
        if self.group != "live":
            return

        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None

        def _summarize_side(raw: Optional[Dict[str, Any]]) -> str:
            if not isinstance(raw, dict) or not raw:
                return "empty"
            qty_val = _as_float(raw.get("qty") or raw.get("positionAmt"))
            entry_val = _as_float(raw.get("entry_price") or raw.get("entryPrice"))
            lev_val = _as_float(raw.get("leverage"))
            upnl_val = _as_float(raw.get("unrealized") or raw.get("unrealizedProfit"))
            parts = []
            if qty_val is not None:
                parts.append(f"qty={qty_val:.8g}")
            if entry_val is not None:
                parts.append(f"entry={entry_val:.8g}")
            if lev_val is not None:
                parts.append(f"lev={lev_val:.8g}")
            if upnl_val is not None:
                parts.append(f"upnl={upnl_val:.8g}")
            entry_order_id = raw.get("entry_order_id") or raw.get("entryOrderId")
            if entry_order_id not in (None, ""):
                parts.append(f"entry_order_id={entry_order_id}")
            return " ".join(parts) if parts else "empty"

        def _summarize_positions(
            positions: Optional[Dict[str, Dict[str, Any]]]
        ) -> Tuple[str, str]:
            if not isinstance(positions, dict):
                return "n/a", "n/a"
            return _summarize_side(positions.get("long")), _summarize_side(positions.get("short"))

        pending_side = "long" if direction > 0 else "short"
        pending = self._pending_exchange_exits.get(pending_side) if self._pending_exchange_exits else None
        pending_summary = "none"
        if isinstance(pending, dict):
            pending_summary = (
                f"qty={pending.get('qty') or 'n/a'} "
                f"positionSide={pending.get('positionSide') or 'n/a'} "
                f"entry_order_id={pending.get('entry_order_id') or 'n/a'} "
                f"entry_price={pending.get('entry_price') or 'n/a'} "
                f"exit_reason={pending.get('exit_reason') or 'n/a'}"
            )

        local_long, local_short = _summarize_positions(
            self.latest_payload.positions if self.latest_payload else None
        )
        exch_long, exch_short = _summarize_positions(exchange_positions or self._exchange_positions)

        logger.warning(
            "Live exit rejection context: symbol=%s reason=%s refresh_idx=%s "
            "side=%s positionSide=%s desired_position_side=%s direction=%s "
            "qty=%.8g remaining_qty=%.8g price=%.8g reduce_only=%s post_only=%s "
            "leverage=%.6g tick_size=%.8g step_size=%.8g attempt=%s "
            "dual_side=%s sync_exchange_positions=%s exchange_position_authority=%s "
            "retry_without_reduce_only=%s "
            "pending_exit=%s local_positions{long=%s short=%s} "
            "exchange_positions{long=%s short=%s} error=%s",
            self.symbol,
            reason,
            "n/a" if refresh_idx is None else refresh_idx,
            side,
            position_side or "n/a",
            desired_position_side or "n/a",
            direction,
            qty,
            remaining_qty,
            price,
            reduce_only,
            use_post_only,
            leverage,
            tick_size,
            step_size,
            attempt,
            self._dual_side_position,
            self.sync_exchange_positions,
            self.exchange_position_authority,
            retry_without_reduce_only,
            pending_summary,
            local_long,
            local_short,
            exch_long,
            exch_short,
            error_detail,
        )

    @staticmethod
    def _timestamp_to_epoch(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            ts = float(value)
        else:
            try:
                ts = float(value)
            except (TypeError, ValueError):
                return None
        if not math.isfinite(ts):
            return None
        if ts > 1e12:
            ts = ts / 1000.0
        elif ts > 1e10:
            ts = ts / 1000.0
        return ts if ts > 0 else None

    @staticmethod
    def _timestamp_to_iso(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            ts = RealtimeRunner._timestamp_to_epoch(value)
            if ts is None:
                return value
        else:
            ts = RealtimeRunner._timestamp_to_epoch(value)
        if ts is None:
            return None
        try:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return None

    @staticmethod
    def _serialize_positions(
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        cleaned: Dict[str, Dict[str, Any]] = {}
        for side in ("long", "short"):
            raw = positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                cleaned[side] = {}
                continue
            entry_time = raw.get("entry_time")
            if isinstance(entry_time, pd.Timestamp):
                entry_time = entry_time.isoformat()
            elif isinstance(entry_time, datetime):
                entry_time = entry_time.isoformat()
            cleaned[side] = {
                "entry_price": raw.get("entry_price"),
                "qty": raw.get("qty"),
                "direction": raw.get("direction"),
                "leverage": raw.get("leverage"),
                "entry_time": entry_time,
                "entry_trade_id": raw.get("entry_trade_id"),
                "entry_order_id": raw.get("entry_order_id"),
            }
        return cleaned

    @staticmethod
    def _reconcile_trades_with_exchange(
        trades: List[Dict[str, Any]],
        exchange_positions: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not trades:
            return trades
        remaining: Dict[str, float] = {}
        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            try:
                qty_val = float(raw.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty_val = 0.0
            remaining[side] = max(qty_val, 0.0)
        reconciled: List[Dict[str, Any]] = []
        for trade in trades:
            trade_type = str(trade.get("type", ""))
            direction = int(trade.get("direction", 0) or 0)
            if direction == 0:
                continue
            side = "long" if direction > 0 else "short"
            opposite = "short" if side == "long" else "long"
            trade_qty = float(trade.get("qty", 0.0) or 0.0)
            if trade_qty <= 0:
                continue
            if trade_type == "entry":
                if remaining.get(side, 0.0) > 0:
                    continue
                if remaining.get(opposite, 0.0) > 0:
                    continue
                reconciled.append(trade)
                continue
            if trade_type in {"exit", "exit_partial"}:
                available = remaining.get(side, 0.0)
                if available <= 0:
                    continue
                adjusted_qty = min(trade_qty, available)
                if adjusted_qty <= 0:
                    continue
                if adjusted_qty != trade_qty:
                    trade = dict(trade)
                    trade["qty"] = adjusted_qty
                remaining[side] = max(available - adjusted_qty, 0.0)
                reconciled.append(trade)
                continue
            reconciled.append(trade)
        return reconciled

    def _maker_price(self, price: float, side: str) -> float:
        if price <= 0:
            return price
        offset = float(self.maker_offset_bps or 0.0) / 10000.0
        if offset <= 0:
            return price
        if side.upper() == "BUY":
            return price * (1.0 - offset)
        return price * (1.0 + offset)

    @staticmethod
    def _normalize_live_order_mode(value: Optional[str]) -> str:
        if not value:
            return "maker"
        text = str(value).strip().lower()
        if text in {"maker", "post", "post_only", "postonly", "limit"}:
            return "maker"
        if text in {"taker", "market", "mkt", "t"}:
            return "taker"
        return "maker"

    def _resolve_live_order_mode(
        self,
        side: str,
        position_side: Optional[str] = None,
        is_exit: Optional[bool] = None,
    ) -> str:
        if is_exit is None:
            is_exit = False
        if is_exit:
            return self.live_order_mode_close or "maker"
        return self.live_order_mode_open or "maker"

    def _apply_maker_aggressive_ticks(
        self, price: float, side: str, tick_size: float
    ) -> float:
        if price <= 0 or tick_size <= 0:
            return price
        ticks = int(self.maker_aggressive_ticks or 0)
        if ticks <= 0:
            return price
        delta = tick_size * ticks
        if side.upper() == "BUY":
            return price + delta
        adjusted = price - delta
        return adjusted if adjusted > 0 else price

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step) * step

    @staticmethod
    def _round_step_with_tolerance(value: float, step: float) -> float:
        if step <= 0:
            return value
        epsilon = step * 1e-9
        return math.floor((value + epsilon) / step) * step

    @staticmethod
    def _precision_from_step(step: float) -> int:
        if step <= 0:
            return 0
        try:
            exp = Decimal(str(step)).as_tuple().exponent
        except Exception:
            return 0
        return max(-exp, 0)

    @classmethod
    def _format_by_step(cls, value: float, step: float) -> str:
        if step <= 0:
            return str(value)
        precision = cls._precision_from_step(step)
        return f"{value:.{precision}f}"

    @staticmethod
    def _round_price(value: float, tick: float, side: str) -> float:
        if tick <= 0:
            return value
        if side.upper() == "BUY":
            return math.floor(value / tick) * tick
        return math.ceil(value / tick) * tick

    @staticmethod
    def _round_price_market(value: float, tick: float, side: str) -> float:
        if tick <= 0:
            return value
        if side.upper() == "BUY":
            return math.ceil(value / tick) * tick
        return math.floor(value / tick) * tick

    @staticmethod
    def _round_price_with_bounds(
        value: float,
        tick: float,
        side: str,
        min_price: Optional[float],
        max_price: Optional[float],
        marketable: bool = False,
    ) -> float:
        if value <= 0:
            return value
        if min_price is not None and value < min_price:
            value = min_price
        if max_price is not None and value > max_price:
            value = max_price
        if tick <= 0:
            return value
        side_up = side.upper()
        if marketable:
            if side_up == "BUY":
                rounded = math.ceil(value / tick) * tick
            else:
                rounded = math.floor(value / tick) * tick
        else:
            if side_up == "BUY":
                rounded = math.floor(value / tick) * tick
            else:
                rounded = math.ceil(value / tick) * tick
        if max_price is not None and rounded > max_price:
            if side_up == "BUY":
                rounded = math.floor(max_price / tick) * tick
            else:
                rounded = math.floor(max_price / tick) * tick
        if min_price is not None and rounded < min_price:
            if side_up == "BUY":
                rounded = math.ceil(min_price / tick) * tick
            else:
                rounded = math.ceil(min_price / tick) * tick
        if min_price is not None and rounded < min_price:
            rounded = min_price
        if max_price is not None and rounded > max_price:
            rounded = max_price
        return rounded

    @staticmethod
    def _price_bounds(
        reference_price: float, filters: Optional[SymbolFilters]
    ) -> Tuple[Optional[float], Optional[float]]:
        if filters is None:
            return None, None
        min_price = filters.min_price
        max_price = filters.max_price
        if reference_price > 0:
            if filters.percent_down is not None and filters.percent_down > 0:
                candidate = reference_price * filters.percent_down
                min_price = candidate if min_price is None else max(min_price, candidate)
            if filters.percent_up is not None and filters.percent_up > 0:
                candidate = reference_price * filters.percent_up
                max_price = candidate if max_price is None else min(max_price, candidate)
        return min_price, max_price
