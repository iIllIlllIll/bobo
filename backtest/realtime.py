from __future__ import annotations

import asyncio
import json
import math
import logging
import re
import time
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


def _is_post_only_reject(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code in {-5022, -2010}:
        return True
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return "post only" in text or "post-only" in text or "immediately trigger" in text


def _is_filter_error(exc: Exception) -> bool:
    code, msg, body = _extract_binance_error(exc)
    if code in {-1013, -4016}:
        return True
    text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
    return (
        "price_filter" in text
        or "lot_size" in text
        or "min_notional" in text
        or "filter" in text
        or "invalid price" in text
        or "invalid quantity" in text
        or "limit price can't be higher" in text
        or "limit price can't be lower" in text
    )


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
    qty = float(trade.get("qty", 0.0) or 0.0)
    entry_reason = str(trade.get("entry_reason") or trade.get("reason") or "")
    exit_reason = str(trade.get("exit_reason", ""))
    entry_time = str(trade.get("entry_time", ""))
    # Exclude price so live candle recalcs don't spam duplicate messages.
    return (trade_type, direction, timestamp, qty, entry_reason, exit_reason, entry_time)


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
        auto_trade: bool = False,
        sync_exchange_positions: bool = False,
        sync_interval_seconds: float = 10.0,
        sync_history_path: Optional[str] = None,
        live_order_retry_seconds: float = 30.0,
        live_order_max_attempts: int = 0,
        dual_side_position: Optional[bool] = None,
        on_trade: Optional[Callable[[str, Dict[str, Any], pd.DataFrame], None]] = None,
        on_payload: Optional[Callable[[str, RunnerPayload], None]] = None,
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
        self.auto_trade = auto_trade
        self.sync_exchange_positions = sync_exchange_positions
        self.sync_interval_seconds = sync_interval_seconds
        self.sync_history_path = sync_history_path
        self.live_order_retry_seconds = live_order_retry_seconds
        self.live_order_max_attempts = live_order_max_attempts
        self._dual_side_position = dual_side_position
        self.on_trade = on_trade
        self.on_payload = on_payload
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._data = pd.DataFrame()
        self._last_close_time: Optional[int] = None
        self._live_row: Optional[pd.Series] = None
        self._seen_trades: set[tuple] = set()
        self._emit_after_ts: Optional[float] = None
        self.latest_payload: Optional[RunnerPayload] = None
        self._symbol_filters: Optional[SymbolFilters] = None
        self._exchange_positions: Optional[Dict[str, Dict[str, Any]]] = None
        self._exchange_balance: Optional[Dict[str, float]] = None
        self._last_exchange_sync: Optional[float] = None
        self._last_sync_snapshot: Optional[str] = None

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
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_balance=self.initial_balance,
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
        exchange_positions: Optional[Dict[str, Dict[str, Any]]] = None
        if self.group == "live" and self.sync_exchange_positions and self.binance:
            exchange_positions = await self._sync_exchange_positions(open_positions)
            if exchange_positions is not None:
                open_positions = exchange_positions
        if emit_trades:
            trades = result.trades
            if exchange_positions is not None:
                trades = self._reconcile_trades_with_exchange(trades, exchange_positions)
            await self._emit_trades(trades, data)
        self._update_payload(result, data, open_positions_override=open_positions)

    async def _emit_trades(self, trades: List[Dict[str, Any]], data: pd.DataFrame) -> None:
        for trade in trades:
            if self._emit_after_ts is not None:
                ts_val = self._trade_timestamp_seconds(trade)
                if ts_val is not None and ts_val < self._emit_after_ts:
                    continue
            key = _trade_key(trade)
            if key in self._seen_trades:
                continue
            self._seen_trades.add(key)
            if self.on_trade:
                self.on_trade(self.group, trade, data)
            if self.auto_trade and self.binance:
                await self._submit_live_order(trade)

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

    async def _submit_live_order(self, trade: Dict[str, Any]) -> None:
        trade_type = trade.get("type")
        if trade_type not in {"entry", "exit", "exit_partial"}:
            return
        if self.binance is None:
            return
        direction = int(trade.get("direction", 0) or 0)
        if direction == 0:
            return
        qty = float(trade.get("qty", 0.0) or 0.0)
        if qty <= 0:
            return
        price = float(trade.get("price", 0.0) or 0.0)
        if price <= 0:
            return

        side = "BUY" if direction > 0 else "SELL"
        reduce_only = False
        position_side = None
        if trade_type in {"exit", "exit_partial"}:
            reduce_only = True
            side = "SELL" if direction > 0 else "BUY"
        dual_side = await self._ensure_dual_side_position()
        if dual_side:
            position_side = "LONG" if direction > 0 else "SHORT"

        if self._symbol_filters is None:
            try:
                self._symbol_filters = await asyncio.to_thread(
                    self.binance.get_symbol_filters, self.symbol
                )
            except Exception:
                self._symbol_filters = None

        tick_size = self._symbol_filters.tick_size if self._symbol_filters else 0.0
        step_size = self._symbol_filters.step_size if self._symbol_filters else 0.0
        remaining_qty = self._round_step(qty, step_size)
        if remaining_qty <= 0:
            return

        attempt = 0
        post_only_rejects = 0
        filters_refreshed = False
        forced_min_price: Optional[float] = None
        forced_max_price: Optional[float] = None
        while remaining_qty > 0:
            attempt += 1
            if self.live_order_max_attempts > 0 and attempt > self.live_order_max_attempts:
                return

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
            adjusted_price = self._maker_price(float(target_price), side)
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
            adjusted_price = self._round_price_with_bounds(
                adjusted_price,
                tick_size,
                side,
                *self._merge_price_bounds(
                    self._price_bounds(float(reference_price), self._symbol_filters),
                    forced_min_price,
                    forced_max_price,
                ),
            )
            if adjusted_price <= 0:
                return

            try:
                order = await asyncio.to_thread(
                    self.binance.place_limit_maker,
                    self.symbol,
                    side,
                    remaining_qty,
                    adjusted_price,
                    reduce_only,
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
                                return
                    except Exception:
                        pass
                    continue
                if _is_post_only_reject(exc):
                    post_only_rejects += 1
                    continue
                return

            if self.live_order_retry_seconds <= 0:
                return

            order_id = None
            if isinstance(order, dict):
                order_id = order.get("orderId")
            if order_id is None:
                return

            await asyncio.sleep(self.live_order_retry_seconds)

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
                return

            if status == "FILLED":
                self._last_exchange_sync = None
                return

            try:
                await asyncio.to_thread(self.binance.cancel_order, self.symbol, order_id)
            except Exception:
                pass

            remaining_qty = max(remaining_qty - executed_qty, 0.0)
            remaining_qty = self._round_step(remaining_qty, step_size)
            self._last_exchange_sync = None

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
            return self._exchange_positions
        exchange_positions = self._parse_exchange_positions(raw_positions)
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
        self._exchange_positions = exchange_positions
        self._last_exchange_sync = time.time()
        self._record_position_sync(engine_positions, exchange_positions)
        return exchange_positions

    def _parse_exchange_positions(
        self, raw_positions: Any
    ) -> Dict[str, Dict[str, Any]]:
        positions: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}
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

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step) * step

    @staticmethod
    def _round_price(value: float, tick: float, side: str) -> float:
        if tick <= 0:
            return value
        if side.upper() == "BUY":
            return math.floor(value / tick) * tick
        return math.ceil(value / tick) * tick

    @staticmethod
    def _round_price_with_bounds(
        value: float,
        tick: float,
        side: str,
        min_price: Optional[float],
        max_price: Optional[float],
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
        if side_up == "BUY":
            rounded = math.floor(value / tick) * tick
        else:
            rounded = math.ceil(value / tick) * tick
        if max_price is not None and rounded > max_price:
            rounded = math.floor(max_price / tick) * tick
        if min_price is not None and rounded < min_price:
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
