from __future__ import annotations

import asyncio
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


def _trade_key(trade: Dict[str, Any]) -> tuple:
    trade_type = str(trade.get("type", ""))
    direction = int(trade.get("direction", 0))
    timestamp = str(trade.get("timestamp"))
    entry_reason = str(trade.get("entry_reason") or trade.get("reason") or "")
    exit_reason = str(trade.get("exit_reason", ""))
    entry_time = str(trade.get("entry_time", ""))
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
        live_order_retry_seconds: float = 20.0,
        live_order_retry_max_seconds: float = 180.0,
        live_order_max_attempts: int = 0,
        dual_side_position: Optional[bool] = None,
        state_path: Optional[str] = None,
        persist_seen_limit: int = 2000,
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
        self.live_order_retry_seconds = live_order_retry_seconds
        self.live_order_retry_max_seconds = live_order_retry_max_seconds
        self.live_order_max_attempts = live_order_max_attempts
        self._dual_side_position = dual_side_position
        self.state_path = state_path
        self._persist_seen_limit = max(int(persist_seen_limit or 0), 100)
        self.on_trade = on_trade
        self.on_payload = on_payload
        self.on_order = on_order
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
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
        self._last_exchange_sync: Optional[float] = None
        self._last_sync_snapshot: Optional[str] = None
        self._exchange_sync_ready = (not self.sync_exchange_positions) or (self.binance is None)
        self._last_sync_guard_warning: Optional[float] = None
        self._target_cache: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        self._entry_cache: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
        self._load_state()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()

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
        await self._bootstrap_data()
        interval_sec = _interval_seconds(self.interval)
        while not self._stopped:
            try:
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
        if self._emit_after_ts is None:
            interval_sec = _interval_seconds(self.interval)
            self._emit_after_ts = time.time() - interval_sec
        await self._run_engine(emit_trades=False)

    async def _update_from_feed(self) -> None:
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
        await self._run_engine()

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
        if self._live_row is not None:
            live_df = self._live_row.to_frame().T.drop(columns=["close_time"], errors="ignore")
            data = pd.concat([data, live_df], ignore_index=True)
            ignore_last_row_signals = True
        if self._should_delay_signals():
            last_closed_index = len(self._data) - 1
            if last_closed_index >= 0:
                suppress_signals_from_index = last_closed_index
        strategy_class = get_strategy_class(self.strategy_name)
        params = _normalize_cheatkey_params(self.strategy_params)
        strategy = strategy_class(**params)
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
            engine.run, None, False, ignore_last_row_signals, suppress_signals_from_index
        )
        open_positions = result.open_positions or {}
        self._update_target_cache(open_positions)
        self._update_entry_cache(open_positions)
        exchange_positions: Optional[Dict[str, Dict[str, Any]]] = None
        if self.group == "live" and self.sync_exchange_positions and self.binance:
            exchange_positions = await self._sync_exchange_positions(open_positions)
            if exchange_positions is not None:
                open_positions = exchange_positions
        if emit_trades:
            trades = result.trades
            if exchange_positions is not None:
                trades = self._reconcile_trades_with_exchange(trades, exchange_positions)
                seeded_trades = self._seed_exchange_exit_trades(
                    result, data, exchange_positions, trades
                )
                if seeded_trades:
                    trades = list(trades) + seeded_trades
            await self._emit_trades(trades, data)
        self._update_payload(result, data, open_positions_override=open_positions)

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
            ts_val = self._trade_timestamp_seconds(trade)
            if self.auto_trade and self.binance:
                if self._should_block_entry_for_sync(trade):
                    self._log_sync_guard_once()
                    continue
                result = await self._submit_live_order(trade)
                if self.on_order:
                    self.on_order(self.group, trade, result)
                if not result.success:
                    self._seen_trades.add(key)
                    self._recent_trade_keys.append(key)
                    state_dirty = True
                    if ts_val is not None:
                        latest_ts = ts_val if latest_ts is None else max(latest_ts, ts_val)
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
                        pass
        try:
            ts_val = float(value)
        except (TypeError, ValueError):
            ts_val = None
        if ts_val is None or not math.isfinite(ts_val):
            return None
        abs_val = abs(ts_val)
        if abs_val >= 1e18:
            ts_val = ts_val / 1e9
        elif abs_val >= 1e15:
            ts_val = ts_val / 1e6
        elif abs_val >= 1e12:
            ts_val = ts_val / 1e3
        return ts_val

    def _update_payload(
        self,
        result: Any,
        data: pd.DataFrame,
        open_positions_override: Optional[Dict[str, Dict[str, Any]]] = None,
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
        positions = self._build_position_payloads(open_positions, price)
        payload = RunnerPayload(
            symbol=self.symbol,
            interval=self.interval,
            price=price,
            equity=equity,
            free_balance=free_balance,
            positions=positions,
            updated_at=time.time(),
        )
        self.latest_payload = payload
        if self.on_payload:
            self.on_payload(self.group, payload)

    def _build_position_payloads(
        self, open_positions: Dict[str, Dict[str, Any]], price: float
    ) -> Dict[str, Dict[str, Any]]:
        payloads: Dict[str, Dict[str, Any]] = {}
        for side in ("long", "short"):
            raw = open_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                payloads[side] = {"active": False}
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

            payloads[side] = {
                "active": True,
                "qty": qty,
                "notional": notional,
                "leverage": leverage,
                "entry_price": entry_price,
                "entry_time": raw.get("entry_time"),
                "direction": direction,
                "tp_pnl": tp_pnl,
                "sl_pnl": sl_pnl,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "pnl_pct": pnl_pct,
                "pnl_amount": pnl_amount,
            }
        return payloads

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
            return None
        if isinstance(payload, dict):
            self._dual_side_position = bool(payload.get("dualSidePosition"))
            return self._dual_side_position
        return None

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
        try:
            raw_positions = await asyncio.to_thread(
                self.binance.get_position_risk, self.symbol
            )
        except Exception:
            return None
        exchange_positions = self._parse_exchange_positions(raw_positions)
        self._exchange_positions = exchange_positions
        self._last_exchange_sync = time.time()
        raw = exchange_positions.get(side, {})
        try:
            qty = float(raw.get("qty", 0.0) or 0.0) if isinstance(raw, dict) else 0.0
        except (TypeError, ValueError):
            qty = 0.0
        return qty if qty > 0 else None

    async def _submit_dust_close(
        self, side: str, position_side: Optional[str], tick_size: float
    ) -> LiveOrderResult:
        if self.binance is None:
            return LiveOrderResult(False, "skipped", "binance client not configured")
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
            return LiveOrderResult(False, "error", detail or str(exc))
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
            return LiveOrderResult(False, "error", detail or str(exc))
        order_id = order.get("orderId") if isinstance(order, dict) else None
        return LiveOrderResult(
            True,
            "submitted",
            "dust close stop-market submitted",
            order_id,
        )

    async def _submit_live_order(self, trade: Dict[str, Any]) -> LiveOrderResult:
        start_ts = time.time()
        attempt = 0

        def _make_result(
            success: bool,
            status: str,
            detail: str = "",
            order_id: Optional[int] = None,
        ) -> LiveOrderResult:
            return LiveOrderResult(
                success,
                status,
                detail,
                order_id,
                attempt,
                time.time() - start_ts,
            )

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
        if trade_type == "entry" and self.sync_exchange_positions:
            if self._has_open_positions():
                return _make_result(
                    False,
                    "blocked",
                    "entry blocked: exchange position already open",
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
            if self._exchange_has_open_position(payload, self.symbol):
                logger.warning(
                    "Live entry blocked: exchange position already open. symbol=%s",
                    self.symbol,
                )
                return _make_result(
                    False,
                    "blocked",
                    "entry blocked: exchange position already open",
                )
        leverage_val = float(trade.get("leverage", self.leverage) or self.leverage or 1.0)
        if leverage_val <= 0:
            leverage_val = 1.0

        if trade_type in {"exit", "exit_partial"}:
            exchange_qty = await self._resolve_exchange_exit_qty(direction)
            if exchange_qty is not None and exchange_qty > 0:
                qty = exchange_qty

        side = "BUY" if direction > 0 else "SELL"
        reduce_only = False
        position_side = None
        if trade_type in {"exit", "exit_partial"}:
            reduce_only = True
            side = "SELL" if direction > 0 else "BUY"
        dual_side = await self._ensure_dual_side_position()
        if dual_side:
            position_side = "LONG" if direction > 0 else "SHORT"
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

        while remaining_qty > 0:
            elapsed = time.time() - start_ts
            force_taker_after_max = (
                reduce_only_param
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

            use_post_only = not force_taker_after_max
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
            if min_notional and not reduce_only_param:
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
                        False,
                        "blocked",
                        (
                            f"min_notional={min_notional:.6g} current="
                            f"{rounded_notional:.6g} symbol={self.symbol} side={side} "
                            f"qty={remaining_qty:.8g} price={adjusted_price:.8g}"
                        ),
                    )

            if not reduce_only_param:
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
                if _is_reduce_only_not_required(exc) and reduce_only_param:
                    reduce_only_param = False
                    continue
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

            skip_fill_check = self.live_order_retry_seconds <= 0
            status_sleep = self.live_order_retry_seconds
            if force_taker_after_max:
                skip_fill_check = False
                status_sleep = 0.0
            if skip_fill_check:
                order_id = order.get("orderId") if isinstance(order, dict) else None
                return _make_result(
                    False,
                    "submitted",
                    "order submitted without fill check",
                    order_id,
                )

            order_id = None
            if isinstance(order, dict):
                order_id = order.get("orderId")
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
                return _make_result(True, "filled", order_id=order_id)

            try:
                await asyncio.to_thread(self.binance.cancel_order, self.symbol, order_id)
            except Exception:
                pass

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
        return _make_result(False, "not_filled", "order loop exited without fill")

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
        if not self.sync_exchange_positions:
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
        exchange_positions = self._parse_exchange_positions(raw_positions)
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
        exchange_positions = self._merge_exchange_positions(
            engine_positions, exchange_positions
        )
        exchange_positions = self._apply_target_cache(exchange_positions)
        exchange_positions = self._apply_entry_cache(exchange_positions)
        self._exchange_positions = exchange_positions
        self._last_exchange_sync = time.time()
        self._record_position_sync(engine_positions, exchange_positions)
        return exchange_positions

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

    def _update_target_cache(self, engine_positions: Dict[str, Dict[str, Any]]) -> None:
        if not isinstance(engine_positions, dict):
            return
        def _as_float(value: Any) -> Optional[float]:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return None
            return value_f if math.isfinite(value_f) else None
        for side in ("long", "short"):
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

        for side in ("long", "short"):
            raw = exchange_positions.get(side, {})
            if not isinstance(raw, dict) or not raw:
                self._target_cache[side] = {}
                continue
            qty_val = _as_float(raw.get("qty", 0.0)) or 0.0
            if qty_val <= 0:
                self._target_cache[side] = {}
                continue
            cache = self._target_cache.get(side) or {}
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
            try:
                qty = float(ex.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue
            if side in existing_exit_sides:
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
                }
            )
        return seeded

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

    def _has_open_positions(self) -> bool:
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

        for side in ("long", "short"):
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
                leverage = float(item.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            qty = abs(pos_amt)
            positions[side] = {
                "entry_price": entry_price,
                "qty": qty,
                "direction": direction,
                "leverage": leverage,
                "entry_time": None,
                "exchange": True,
            }
        return positions

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
