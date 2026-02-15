from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from urllib.parse import urlencode


@dataclass
class SymbolFilters:
    tick_size: float
    step_size: float


class BinanceFuturesClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://fapi.binance.com",
        recv_window: int = 5000,
        timeout: int = 10,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.timeout = timeout
        self._symbol_filters: Dict[str, SymbolFilters] = {}

    def _sign(self, params: list[tuple[str, Any]] | Dict[str, Any]) -> str:
        if isinstance(params, dict):
            query = urlencode(params, doseq=True)
        else:
            query = urlencode(params, doseq=True)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        base_params = dict(params or {})
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        request_params: Dict[str, Any] | list[tuple[str, Any]]
        if signed:
            param_items = list(base_params.items())
            param_items.append(("timestamp", int(time.time() * 1000)))
            param_items.append(("recvWindow", self.recv_window))
            signature = self._sign(param_items)
            param_items.append(("signature", signature))
            request_params = param_items
        else:
            request_params = base_params
        url = f"{self.base_url}{path}"
        resp = requests.request(
            method,
            url,
            params=request_params,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_price(self, symbol: str) -> float:
        data = self._request(
            "GET",
            "/fapi/v1/ticker/price",
            params={"symbol": symbol.upper()},
        )
        return float(data["price"])

    def get_klines(self, symbol: str, interval: str, limit: int = 2) -> list[list]:
        return self._request(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        )

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol.upper(), "leverage": leverage},
            signed=True,
        )

    def get_account(self) -> Any:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_position_risk(self, symbol: Optional[str] = None) -> Any:
        params = {"symbol": symbol.upper()} if symbol else None
        return self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)

    def get_exchange_info(self) -> Any:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        symbol = symbol.upper()
        cached = self._symbol_filters.get(symbol)
        if cached:
            return cached
        info = self.get_exchange_info()
        for entry in info.get("symbols", []):
            if entry.get("symbol") != symbol:
                continue
            filters = {item.get("filterType"): item for item in entry.get("filters", [])}
            price_filter = filters.get("PRICE_FILTER") or {}
            lot_filter = filters.get("LOT_SIZE") or {}
            tick = float(price_filter.get("tickSize", 0.0))
            step = float(lot_filter.get("stepSize", 0.0))
            if tick <= 0:
                tick = 0.01
            if step <= 0:
                step = 0.001
            parsed = SymbolFilters(tick_size=tick, step_size=step)
            self._symbol_filters[symbol] = parsed
            return parsed
        raise ValueError(f"Symbol not found in exchange info: {symbol}")

    def place_limit_maker(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool = False,
    ) -> Any:
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT_MAKER",
            "quantity": qty,
            "price": price,
            "reduceOnly": "true" if reduce_only else "false",
        }
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> Any:
        if order_id is None and not orig_client_order_id:
            raise ValueError("order_id or orig_client_order_id is required")
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self._request("GET", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> Any:
        if order_id is None and not orig_client_order_id:
            raise ValueError("order_id or orig_client_order_id is required")
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)
