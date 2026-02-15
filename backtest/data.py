from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import requests


def load_price_data(
    folder: str | Path,
    file_pattern: str = "*.csv",
    timestamp_column: str = "timestamp",
    timezone: Optional[str] = "UTC",
    datetime_format: Optional[str] = None,
) -> pd.DataFrame:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Price data folder not found: {folder_path}")

    files = sorted(folder_path.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No price data files found in {folder_path} matching {file_pattern}"
        )

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        if timestamp_column not in df.columns:
            raise ValueError(f"Missing timestamp column '{timestamp_column}' in {fp}")
        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column], format=datetime_format, utc=True
        )
        if timezone:
            df[timestamp_column] = df[timestamp_column].dt.tz_convert(timezone)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(timestamp_column).reset_index(drop=True)
    data = data.rename(columns={timestamp_column: "timestamp"})
    return data


_BINANCE_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


def _to_utc_ms(value: str) -> int:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid datetime: {value}")
    return int(ts.value // 1_000_000)


def _binance_endpoint(market: str) -> str:
    if market == "spot":
        return "https://api.binance.com/api/v3/klines"
    if market == "futures":
        return "https://fapi.binance.com/fapi/v1/klines"
    raise ValueError("market must be 'spot' or 'futures'")


def _fetch_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market: str = "futures",
    limit: int = 1000,
) -> list[list]:
    url = _binance_endpoint(market)
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _klines_to_frame(rows: Iterable[Iterable]) -> pd.DataFrame:
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
            }
        )
    return pd.DataFrame(data)


def download_price_data(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    output_dir: str | Path = "price data",
    market: str = "futures",
) -> Path:
    if interval not in _BINANCE_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'.")

    start_ms = _to_utc_ms(start)
    end_ms = _to_utc_ms(end)
    if start_ms >= end_ms:
        raise ValueError("start must be earlier than end")

    all_rows: list[list] = []
    next_start = start_ms
    while next_start < end_ms:
        rows = _fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            start_ms=next_start,
            end_ms=end_ms,
            market=market,
        )
        if not rows:
            break
        all_rows.extend(rows)
        last_open = int(rows[-1][0])
        if last_open <= next_start:
            break
        next_start = last_open + 1

    if not all_rows:
        raise ValueError("No data returned for the requested range.")

    df = _klines_to_frame(all_rows)
    df = df[df["timestamp"] <= pd.to_datetime(end_ms, unit="ms", utc=True)]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.upper().replace("/", "_")
    start_label = pd.to_datetime(start, utc=True).strftime("%Y%m%d")
    end_label = pd.to_datetime(end, utc=True).strftime("%Y%m%d")
    filename = f"{safe_symbol}_{interval}_{start_label}_{end_label}.csv"
    file_path = output_path / filename
    df.to_csv(file_path, index=False)
    return file_path
