from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Any, Dict, Callable

import matplotlib
import numpy as np
import pandas as pd
import warnings

from backtest.result import BacktestResult

# Suppress noisy font glyph warnings when non-ASCII labels are present.
warnings.filterwarnings(
    "ignore",
    message=r"Glyph .* missing from font\(s\).*",
    category=UserWarning,
)


def create_backtest_chart(result: BacktestResult, output_path: Path) -> Optional[Path]:
    if result.equity_curve.empty:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result.equity_curve.copy()
    equity = df["equity"].astype(float)
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0

    if "timestamp" in df.columns:
        x_values = pd.to_datetime(df["timestamp"], errors="ignore")
    else:
        x_values = df.index

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(x_values, equity, color="#1f77b4", linewidth=1.6)
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.3)

    drawdown_pct = drawdown * 100.0
    axes[1].plot(x_values, drawdown_pct, color="#d62728", linewidth=1.2)
    axes[1].fill_between(x_values, drawdown_pct, 0, color="#d62728", alpha=0.2)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_costom_loss_chart(
    reason: str,
    losses: list[float],
    output_path: Path,
    max_points: int = 200,
) -> Optional[Path]:
    if not losses:
        return None

    sorted_losses = sorted(losses, key=lambda value: abs(value), reverse=True)
    if max_points and len(sorted_losses) > max_points:
        sorted_losses = sorted_losses[:max_points]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = list(range(1, len(sorted_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x_values, sorted_losses, color="#1f77b4", alpha=0.85)
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    mean_val = float(np.mean(sorted_losses))
    median_val = float(np.median(sorted_losses))
    ax.axhline(mean_val, color="#004c99", linewidth=1.3, alpha=0.85, label="Mean")
    ax.axhline(median_val, color="#66a3ff", linewidth=1.3, alpha=0.85, label="Median")
    ax.set_title(f"Costom breach loss% (leveraged) - {reason}")
    ax.set_xlabel("Trade (sorted by |loss|)")
    ax.set_ylabel("Return %")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_equity_price_chart(
    result: BacktestResult, data: pd.DataFrame, output_path: Path
) -> Optional[Path]:
    if result.equity_curve.empty or data.empty:
        return None
    if "timestamp" not in result.equity_curve.columns:
        return None
    if "timestamp" not in data.columns or "close" not in data.columns:
        return None

    equity_df = result.equity_curve[["timestamp", "equity"]].copy()
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(equity_df["timestamp"]):
        equity_df["timestamp"] = equity_df["timestamp"].dt.tz_localize(None)
    equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if equity_df.empty:
        return None

    has_volume = "volume" in data.columns
    price_columns = ["timestamp", "close"] + (["volume"] if has_volume else [])
    price_df = data[price_columns].copy()
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(price_df["timestamp"]):
        price_df["timestamp"] = price_df["timestamp"].dt.tz_localize(None)
    price_df = price_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if price_df.empty:
        return None

    merged = pd.merge_asof(
        equity_df,
        price_df,
        on="timestamp",
        direction="nearest",
    )
    if merged.empty:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if has_volume:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax1 = axes[0]
        ax2 = ax1.twinx()
        ax_vol = axes[1]
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax_vol = None

    ax1.plot(
        merged["timestamp"],
        merged["equity"].astype(float),
        color="#1f77b4",
        linewidth=1.5,
        label="Equity",
    )
    ax2.plot(
        merged["timestamp"],
        merged["close"].astype(float),
        color="#ff7f0e",
        linewidth=1.2,
        alpha=0.85,
        label="Price",
    )

    ax1.set_ylabel("Equity")
    ax2.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Equity vs Price" + (" (with Volume)" if has_volume else ""))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="best")

    if has_volume and ax_vol is not None:
        volumes = merged["volume"].astype(float)
        ax_vol.plot(
            merged["timestamp"],
            volumes,
            color="#7f7f7f",
            linewidth=1.0,
            alpha=0.8,
            label="Volume",
        )
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_equity_price_mdd_chart(
    result: BacktestResult,
    data: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    if result.equity_curve.empty or data.empty:
        return None
    if "timestamp" not in result.equity_curve.columns:
        return None
    if "timestamp" not in data.columns or "close" not in data.columns:
        return None

    equity_df = result.equity_curve[["timestamp", "equity"]].copy()
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], errors="coerce")
    equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if equity_df.empty:
        return None

    price_df = data[["timestamp", "close"]].copy()
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(price_df["timestamp"]):
        price_df["timestamp"] = price_df["timestamp"].dt.tz_localize(None)
    price_df = price_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if price_df.empty:
        return None

    merged = pd.merge_asof(
        equity_df,
        price_df,
        on="timestamp",
        direction="nearest",
    )
    if merged.empty:
        return None

    merged["equity"] = merged["equity"].astype(float)
    peak = merged["equity"].cummax()
    drawdown_pct = (merged["equity"] / peak - 1.0) * 100.0

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1 = axes[0]
    ax2 = ax1.twinx()

    ax1.plot(
        merged["timestamp"],
        merged["equity"],
        color="#1f77b4",
        linewidth=1.5,
        label="Equity",
    )
    ax2.plot(
        merged["timestamp"],
        merged["close"].astype(float),
        color="#ff7f0e",
        linewidth=1.2,
        alpha=0.85,
        label="Price",
    )
    ax1.set_ylabel("Equity")
    ax2.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or "Equity vs Price")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="best")

    axes[1].plot(
        merged["timestamp"],
        drawdown_pct,
        color="#d62728",
        linewidth=1.2,
    )
    axes[1].fill_between(merged["timestamp"], drawdown_pct, 0, color="#d62728", alpha=0.2)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_equity_mdd_monthly_stats_chart(
    result: BacktestResult,
    output_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    if result.equity_curve.empty:
        return None
    if "timestamp" not in result.equity_curve.columns:
        return None

    equity_df = result.equity_curve[["timestamp", "equity"]].copy()
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], errors="coerce")
    equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if equity_df.empty:
        return None

    equity = equity_df["equity"].astype(float)
    peak = equity.cummax()
    drawdown_pct = (equity / peak - 1.0) * 100.0

    exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
    trade_rows = []
    for trade in exit_trades:
        exit_time = trade.get("exit_time") or trade.get("timestamp") or trade.get("entry_time")
        exit_ts = pd.to_datetime(exit_time, errors="coerce")
        if pd.isna(exit_ts):
            continue
        try:
            trade_return = float(trade.get("return", 0.0)) * 100.0
        except (TypeError, ValueError):
            trade_return = 0.0
        trade_rows.append({"exit_time": exit_ts, "return": trade_return})

    monthly_df = pd.DataFrame(trade_rows)
    if not monthly_df.empty and pd.api.types.is_datetime64tz_dtype(monthly_df["exit_time"]):
        monthly_df["exit_time"] = monthly_df["exit_time"].dt.tz_localize(None)
    equity_ts = equity_df["timestamp"]
    equity_ts_naive = equity_ts
    if pd.api.types.is_datetime64tz_dtype(equity_ts):
        equity_ts_naive = equity_ts.dt.tz_localize(None)
    stats_rows = []
    month_start = equity_ts_naive.iloc[0].to_period("M")
    month_end = equity_ts_naive.iloc[-1].to_period("M")
    month_index = pd.period_range(start=month_start, end=month_end, freq="M")
    if not monthly_df.empty:
        monthly_df["month"] = monthly_df["exit_time"].dt.to_period("M")
    for month in month_index:
        if not monthly_df.empty:
            group = monthly_df.loc[monthly_df["month"] == month]
        else:
            group = monthly_df
        returns = group["return"].astype(float) if not group.empty else pd.Series([], dtype=float)
        trade_count = int(returns.count())
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_return = float(returns.mean()) if trade_count else np.nan
        win_rate = float(len(wins) / trade_count * 100.0) if trade_count else np.nan
        avg_win = float(wins.mean()) if len(wins) else np.nan
        avg_loss = float(losses.mean()) if len(losses) else np.nan
        stats_rows.append(
            {
                "month": str(month),
                "avg_return": avg_return,
                "win_rate": win_rate,
                "trades": trade_count,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }
        )

    def _fmt_pct(value: Any) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(val):
            return "-"
        return f"{val:.2f}%"

    def _fmt_int(value: Any) -> str:
        try:
            return f"{int(value)}"
        except (TypeError, ValueError):
            return "0"

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_equity = axes[0]
    ax_equity.plot(equity_df["timestamp"], equity, color="#1f77b4", linewidth=1.5)
    equity_min = float(np.nanmin(equity)) if len(equity) else 0.0
    equity_max = float(np.nanmax(equity)) if len(equity) else 0.0
    equity_range = equity_max - equity_min
    if np.isfinite(equity_min) and np.isfinite(equity_max):
        bottom_pad = equity_range * 0.18 if equity_range > 0 else max(1.0, abs(equity_max) * 0.05)
        top_pad = equity_range * 0.02 if equity_range > 0 else 0.0
        ax_equity.set_ylim(equity_min - bottom_pad, equity_max + top_pad)
    ax_equity.set_ylabel("Equity")
    ax_equity.grid(True, alpha=0.3)
    ax_equity.set_title(title or "Equity and Monthly Stats")

    axes[1].plot(
        equity_df["timestamp"],
        drawdown_pct,
        color="#d62728",
        linewidth=1.2,
    )
    axes[1].fill_between(
        equity_df["timestamp"],
        drawdown_pct,
        0,
        color="#d62728",
        alpha=0.2,
    )
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, alpha=0.3)

    if stats_rows:
        month_periods = pd.period_range(
            start=equity_ts_naive.iloc[0].to_period("M"),
            end=equity_ts_naive.iloc[-1].to_period("M"),
            freq="M",
        )
        month_map = {row["month"]: row for row in stats_rows}
        trans = blended_transform_factory(ax_equity.transData, ax_equity.transAxes)
        for idx, month in enumerate(month_periods):
            start_ts = month.start_time
            end_ts = month.end_time
            ax_equity.axvspan(
                start_ts,
                end_ts,
                color="#f0f0f0" if idx % 2 == 0 else "#e8f2ff",
                alpha=0.35,
                linewidth=0.0,
            )
            row = month_map.get(str(month))
            if row is None:
                continue
            avg_return = _fmt_pct(row["avg_return"])
            win_rate = _fmt_pct(row["win_rate"])
            trades = _fmt_int(row["trades"])
            avg_win = _fmt_pct(row["avg_win"])
            avg_loss = _fmt_pct(row["avg_loss"])
            lines = [
                f"{month}",
                f"R  {avg_return:>8}",
                f"W  {win_rate:>8}",
                f"T  {trades:>8}",
                f"W+ {avg_win:>8}",
                f"L- {avg_loss:>8}",
            ]
            text = "\n".join(lines)
            mid_ts = start_ts + (end_ts - start_ts) / 2
            ax_equity.text(
                mid_ts,
                0.03,
                text,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#2c2c2c",
                family="monospace",
                multialignment="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def compute_monthly_return_stats(
    result: BacktestResult,
) -> tuple[pd.DataFrame, dict]:
    if result.equity_curve.empty or "timestamp" not in result.equity_curve.columns:
        empty_df = pd.DataFrame(columns=["month", "start_equity", "end_equity", "return_pct"])
        return empty_df, {"months": 0}

    equity_df = result.equity_curve[["timestamp", "equity"]].copy()
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], errors="coerce")
    equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if equity_df.empty:
        empty_df = pd.DataFrame(columns=["month", "start_equity", "end_equity", "return_pct"])
        return empty_df, {"months": 0}

    if pd.api.types.is_datetime64tz_dtype(equity_df["timestamp"]):
        equity_df["timestamp"] = equity_df["timestamp"].dt.tz_localize(None)

    equity_df["month"] = equity_df["timestamp"].dt.to_period("M")
    month_index = pd.period_range(
        start=equity_df["month"].iloc[0],
        end=equity_df["month"].iloc[-1],
        freq="M",
    )

    rows: list[dict[str, Any]] = []
    grouped = equity_df.groupby("month", sort=True)
    for month in month_index:
        group = grouped.get_group(month) if month in grouped.groups else None
        if group is None or group.empty:
            rows.append(
                {
                    "month": str(month),
                    "start_equity": np.nan,
                    "end_equity": np.nan,
                    "return_pct": np.nan,
                }
            )
            continue
        start_equity = float(group["equity"].iloc[0])
        end_equity = float(group["equity"].iloc[-1])
        if start_equity == 0:
            ret = np.nan
        else:
            ret = (end_equity - start_equity) / start_equity * 100.0
        rows.append(
            {
                "month": str(month),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return_pct": ret,
            }
        )

    monthly_df = pd.DataFrame(rows)
    returns = monthly_df["return_pct"].astype(float)
    returns = returns[np.isfinite(returns)]

    if returns.empty:
        stats = {"months": 0}
    else:
        median = float(np.median(returns))
        mad = float(np.median(np.abs(returns - median)))
        q1 = float(np.percentile(returns, 25))
        q3 = float(np.percentile(returns, 75))
        stats = {
            "months": int(len(returns)),
            "mean": float(np.mean(returns)),
            "median": median,
            "std": float(np.std(returns, ddof=0)),
            "mad": mad,
            "iqr": q3 - q1,
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "pos_rate": float((returns > 0).mean() * 100.0),
        }
        mean = stats["mean"]
        if mean != 0 and np.isfinite(mean):
            stats["cv"] = stats["std"] / abs(mean)
        else:
            stats["cv"] = np.nan

    return monthly_df, stats


def compute_entry_minute_win_rate(
    result: BacktestResult,
) -> tuple[pd.DataFrame, dict]:
    exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
    if not exit_trades:
        empty_df = pd.DataFrame(
            columns=["minute", "label", "wins", "losses", "total", "win_rate"]
        )
        return empty_df, {"trades": 0, "minutes": 0}

    rows: list[dict[str, Any]] = []
    for trade in exit_trades:
        entry_time = trade.get("entry_time")
        if entry_time is None:
            continue
        ts = pd.to_datetime(entry_time, errors="coerce")
        if pd.isna(ts):
            continue
        if isinstance(ts, pd.Timestamp) and ts.tz is not None:
            ts = ts.tz_localize(None)
        minute = int(ts.minute)
        pnl = trade.get("pnl")
        if pnl is None:
            ret = trade.get("return")
            if ret is None:
                continue
            win = float(ret) > 0.0
        else:
            win = float(pnl) > 0.0
        rows.append({"minute": minute, "win": win})

    if not rows:
        empty_df = pd.DataFrame(
            columns=["minute", "label", "wins", "losses", "total", "win_rate"]
        )
        return empty_df, {"trades": 0, "minutes": 0}

    df = pd.DataFrame(rows)
    grouped = df.groupby("minute", sort=True)
    out_rows: list[dict[str, Any]] = []
    for minute, group in grouped:
        total = int(len(group))
        wins = int(group["win"].sum())
        losses = total - wins
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        out_rows.append(
            {
                "minute": int(minute),
                "label": f"{int(minute):02d}",
                "wins": wins,
                "losses": losses,
                "total": total,
                "win_rate": win_rate,
            }
        )

    minute_df = pd.DataFrame(out_rows).sort_values("minute")
    total_trades = int(minute_df["total"].sum()) if not minute_df.empty else 0
    overall_win_rate = (
        float(minute_df["wins"].sum()) / float(total_trades) * 100.0 if total_trades > 0 else 0.0
    )
    stats = {"trades": total_trades, "minutes": int(len(minute_df)), "win_rate": overall_win_rate}
    return minute_df, stats


def _is_tp_reason(reason: Any) -> bool:
    if reason is None:
        return False
    text = str(reason).lower()
    return "take_profit" in text or "rr_tp" in text


def compute_entry_minute_tp_rate(
    result: BacktestResult,
) -> tuple[pd.DataFrame, dict]:
    exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
    total_exit_trades = int(len(exit_trades))
    tp_hits_all = int(
        sum(1 for trade in exit_trades if _is_tp_reason(trade.get("exit_reason")))
    )
    overall_tp_rate = (
        float(tp_hits_all) / float(total_exit_trades) * 100.0
        if total_exit_trades > 0
        else 0.0
    )
    if not exit_trades:
        empty_df = pd.DataFrame(
            columns=["minute", "label", "tp_hits", "total", "tp_rate"]
        )
        return empty_df, {
            "trades": 0,
            "trades_with_time": 0,
            "minutes": 0,
            "tp_rate": 0.0,
        }

    rows: list[dict[str, Any]] = []
    for trade in exit_trades:
        entry_time = trade.get("entry_time")
        if entry_time is None:
            continue
        ts = pd.to_datetime(entry_time, errors="coerce")
        if pd.isna(ts):
            continue
        if isinstance(ts, pd.Timestamp) and ts.tz is not None:
            ts = ts.tz_localize(None)
        minute = int(ts.minute)
        reason = trade.get("exit_reason")
        tp_hit = _is_tp_reason(reason)
        rows.append({"minute": minute, "tp_hit": tp_hit})

    if not rows:
        empty_df = pd.DataFrame(
            columns=["minute", "label", "tp_hits", "total", "tp_rate"]
        )
        return empty_df, {
            "trades": total_exit_trades,
            "trades_with_time": 0,
            "minutes": 0,
            "tp_rate": overall_tp_rate,
        }

    df = pd.DataFrame(rows)
    grouped = df.groupby("minute", sort=True)
    out_rows: list[dict[str, Any]] = []
    for minute, group in grouped:
        total = int(len(group))
        tp_hits = int(group["tp_hit"].sum())
        tp_rate = (tp_hits / total * 100.0) if total > 0 else 0.0
        out_rows.append(
            {
                "minute": int(minute),
                "label": f"{int(minute):02d}",
                "tp_hits": tp_hits,
                "total": total,
                "tp_rate": tp_rate,
            }
        )

    minute_df = pd.DataFrame(out_rows).sort_values("minute")
    covered_trades = int(minute_df["total"].sum()) if not minute_df.empty else 0
    stats = {
        "trades": total_exit_trades,
        "trades_with_time": covered_trades,
        "minutes": int(len(minute_df)),
        "tp_rate": overall_tp_rate,
    }
    return minute_df, stats


def create_monthly_return_stability_chart(
    result: BacktestResult,
    output_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    monthly_df, stats = compute_monthly_return_stats(result)
    if monthly_df.empty:
        return None
    returns = monthly_df["return_pct"].astype(float).to_numpy()
    months = monthly_df["month"].astype(str).to_list()
    if len(months) == 0:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(months))
    colors = []
    for value in returns:
        if not np.isfinite(value):
            colors.append("#7f7f7f")
        elif value >= 0:
            colors.append("#2ca02c")
        else:
            colors.append("#d62728")

    fig_width = 10.5 if len(months) <= 18 else min(16.0, 9.5 + len(months) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar(x, np.nan_to_num(returns, nan=0.0), color=colors, alpha=0.85)

    mean = stats.get("mean")
    std = stats.get("std")
    if mean is not None and np.isfinite(mean):
        ax.axhline(mean, color="#1f77b4", linewidth=1.3, linestyle="--", label="Mean")
        if std is not None and np.isfinite(std):
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color="#1f77b4",
                alpha=0.12,
                label="±1σ",
            )

    ax.set_title(title or "Monthly return stability")
    ax.set_ylabel("Monthly return (%)")
    ax.grid(True, axis="y", alpha=0.3)

    tick_step = max(1, len(months) // 12)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([months[idx] for idx in range(0, len(months), tick_step)], rotation=45, ha="right")

    stats_lines = [
        f"Months  {stats.get('months', 0)}",
    ]
    if stats.get("months", 0) > 0:
        stats_lines += [
            f"Mean    {stats.get('mean', 0.0):.2f}%",
            f"Median  {stats.get('median', 0.0):.2f}%",
            f"Std     {stats.get('std', 0.0):.2f}%",
            f"MAD     {stats.get('mad', 0.0):.2f}%",
            f"IQR     {stats.get('iqr', 0.0):.2f}%",
            f"Min/Max {stats.get('min', 0.0):.2f}% / {stats.get('max', 0.0):.2f}%",
            f"+Rate   {stats.get('pos_rate', 0.0):.1f}%",
        ]
        if np.isfinite(stats.get("cv", np.nan)):
            stats_lines.append(f"CV      {stats.get('cv', 0.0):.2f}")

    ax.text(
        0.01,
        0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_entry_minute_win_rate_chart(
    result: BacktestResult,
    output_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    minute_df, stats = compute_entry_minute_win_rate(result)
    if minute_df.empty:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = minute_df["label"].astype(str).to_list()
    win_rates = minute_df["win_rate"].astype(float).to_list()
    totals = minute_df["total"].astype(int).to_list()

    x = np.arange(len(labels))
    colors = ["#2ca02c" if rate >= 50.0 else "#d62728" for rate in win_rates]

    fig_width = 9.5 if len(labels) <= 16 else min(15.0, 7.5 + len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bars = ax.bar(x, win_rates, color=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_title(title or "Entry minute win rate")
    ax.set_ylabel("Win rate (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        total = totals[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(100, height + 2.0),
            f"{height:.1f}%\nT {total}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2c2c2c",
        )

    stats_lines = [
        f"Trades  {stats.get('trades', 0)}",
        f"Minutes {stats.get('minutes', 0)}",
        f"WinRate {stats.get('win_rate', 0.0):.1f}%",
    ]
    ax.text(
        0.01,
        0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_entry_minute_tp_rate_chart(
    result: BacktestResult,
    output_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    minute_df, stats = compute_entry_minute_tp_rate(result)
    if minute_df.empty:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = minute_df["label"].astype(str).to_list()
    tp_rates = minute_df["tp_rate"].astype(float).to_list()
    totals = minute_df["total"].astype(int).to_list()

    x = np.arange(len(labels))
    colors = ["#1f77b4" if rate >= 50.0 else "#ff7f0e" for rate in tp_rates]

    fig_width = 9.5 if len(labels) <= 16 else min(15.0, 7.5 + len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bars = ax.bar(x, tp_rates, color=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_title(title or "Entry minute TP reach rate")
    ax.set_ylabel("TP reach rate (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        total = totals[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(100, height + 2.0),
            f"{height:.1f}%\nT {total}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2c2c2c",
        )

    trades = int(stats.get("trades", 0) or 0)
    covered = int(stats.get("trades_with_time", trades) or trades)
    stats_lines = [
        f"Trades  {trades} (cov {covered})",
        f"Minutes {stats.get('minutes', 0)}",
        f"TPRate  {stats.get('tp_rate', 0.0):.1f}%",
    ]
    ax.text(
        0.01,
        0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_equity_overlay_monthly_best_chart(
    series: list[dict],
    output_path: Path,
    title: Optional[str] = None,
) -> Optional[Path]:
    curves: list[dict] = []
    for entry in series:
        result = entry.get("result")
        if result is None or result.equity_curve.empty:
            continue
        if "timestamp" not in result.equity_curve.columns:
            continue
        label = str(entry.get("label") or entry.get("value") or "")
        df = result.equity_curve[["timestamp", "equity"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
        if df.empty:
            continue
        if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        curves.append({"label": label, "df": df})

    if not curves:
        return None

    min_ts = min(curve["df"]["timestamp"].iloc[0] for curve in curves)
    max_ts = max(curve["df"]["timestamp"].iloc[-1] for curve in curves)
    month_periods = pd.period_range(start=min_ts.to_period("M"), end=max_ts.to_period("M"), freq="M")

    for curve in curves:
        df = curve["df"].copy()
        df["month"] = df["timestamp"].dt.to_period("M")
        month_returns: dict[str, float] = {}
        for month, group in df.groupby("month"):
            if group.empty:
                continue
            first = float(group["equity"].iloc[0])
            last = float(group["equity"].iloc[-1])
            if first == 0:
                ret = np.nan
            else:
                ret = (last - first) / first * 100.0
            month_returns[str(month)] = ret
        curve["month_returns"] = month_returns

    best_rows: list[dict] = []
    for month in month_periods:
        month_key = str(month)
        best_label = None
        best_return = None
        for curve in curves:
            ret = curve["month_returns"].get(month_key)
            if ret is None or not np.isfinite(ret):
                continue
            if best_return is None or ret > best_return:
                best_return = ret
                best_label = curve["label"]
        best_rows.append({"month": month_key, "label": best_label, "return": best_return})

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for curve in curves:
        ax.plot(
            curve["df"]["timestamp"],
            curve["df"]["equity"].astype(float),
            linewidth=1.2,
            alpha=0.9,
            label=str(curve["label"]),
        )

    all_equity = np.concatenate(
        [curve["df"]["equity"].astype(float).to_numpy() for curve in curves]
    )
    equity_min = float(np.nanmin(all_equity)) if all_equity.size else 0.0
    equity_max = float(np.nanmax(all_equity)) if all_equity.size else 0.0
    equity_range = equity_max - equity_min
    if np.isfinite(equity_min) and np.isfinite(equity_max):
        bottom_pad = equity_range * 0.26 if equity_range > 0 else max(1.0, abs(equity_max) * 0.08)
        top_pad = equity_range * 0.03 if equity_range > 0 else 0.0
        ax.set_ylim(equity_min - bottom_pad, equity_max + top_pad)

    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Equity overlay")
    if len(curves) <= 20:
        ax.legend(fontsize=7, loc="best")
    else:
        ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.subplots_adjust(right=0.78)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    import matplotlib.dates as mdates
    from matplotlib.transforms import Bbox, TransformedBbox
    label_width = 10
    ret_width = 9
    y0 = 0.02
    y_height = 0.18
    for idx, month in enumerate(month_periods):
        start_ts = month.start_time
        end_ts = month.end_time
        ax.axvspan(
            start_ts,
            end_ts,
            color="#f0f0f0" if idx % 2 == 0 else "#e8f2ff",
            alpha=0.35,
            linewidth=0.0,
        )
        row = best_rows[idx] if idx < len(best_rows) else None
        if row is None or row.get("label") is None or row.get("return") is None:
            continue
        label = str(row["label"])
        if len(label) > label_width:
            label = f"{label[: label_width - 2]}.."
        ret = row["return"]
        ret_text = f"{ret:.2f}%" if np.isfinite(ret) else "-"
        lines = [
            f"{month}",
            f"Best {label:<{label_width}}",
            f"Ret  {ret_text:>{ret_width}}",
        ]
        text = "\n".join(lines)
        month_span = end_ts - start_ts
        x_text = start_ts + month_span * 0.03
        text_artist = ax.text(
            x_text,
            y0 + 0.01,
            text,
            transform=trans,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#2c2c2c",
            family="monospace",
            multialignment="left",
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
        )
        x0 = mdates.date2num(start_ts)
        x1 = mdates.date2num(end_ts)
        clip_bbox = Bbox.from_extents(x0, y0, x1, y0 + y_height)
        text_artist.set_clip_box(TransformedBbox(clip_bbox, trans))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_grid_heatmap(
    values1: list[Any],
    values2: list[Any],
    matrix: list[list[Optional[float]]],
    output_path: Path,
    title: Optional[str] = None,
    label1: str = "Var 1",
    label2: str = "Var 2",
    show_values: bool = True,
    colorbar_label: str = "Return %",
    value_formatter: Optional[Callable[[float], str]] = None,
    max_abs: Optional[float] = None,
    value_fontsize: Optional[float] = None,
) -> Optional[Path]:
    if not values1 or not values2:
        return None
    flat_values = [value for row in matrix for value in row if value is not None]
    if not flat_values:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.patches import FancyBboxPatch

    rows = len(values1)
    cols = len(values2)

    width = max(4.5, cols * 0.6 + 2.5)
    height = max(4.0, rows * 0.6 + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f7")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))

    def _blend(start: str, end: str, t: float) -> str:
        r1, g1, b1 = _hex_to_rgb(start)
        r2, g2, b2 = _hex_to_rgb(end)
        r = r1 + (r2 - r1) * t
        g = g1 + (g2 - g1) * t
        b = b1 + (b2 - b1) * t
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    finite_values = [float(value) for value in flat_values if np.isfinite(value)]
    if max_abs is None:
        max_abs = max((abs(value) for value in finite_values), default=0.0)
    pos_light = "#e0f2f1"
    pos_dark = "#26a69a"
    neg_light = "#ffebee"
    neg_dark = "#ef5350"
    neutral = "#dcdcdc"
    missing = "#f2f2f2"

    def cell_color(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return missing
        if value == 0.0:
            return neutral
        norm = abs(value) / max_abs if max_abs > 0.0 else 0.0
        norm = min(1.0, max(0.0, norm))
        t = 0.1 + 0.9 * norm
        if value < 0.0:
            return _blend(neg_light, neg_dark, t)
        return _blend(pos_light, pos_dark, t)

    def text_color_for(face_hex: str, value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "#666666"
        if value == 0.0:
            return "#333333"
        return "white"

    if value_formatter is None:
        value_formatter = lambda value: f"{value:+.2f}%"

    gap = 0.1
    size = 1.0 - gap
    rounding = 0.12

    if value_fontsize is None:
        max_dim = max(rows, cols)
        if max_dim <= 12:
            value_fontsize = 9.5
        elif max_dim <= 16:
            value_fontsize = 9.0
        elif max_dim <= 20:
            value_fontsize = 8.2
        elif max_dim <= 24:
            value_fontsize = 7.5
        else:
            value_fontsize = 6.8

    max_label = ""
    if show_values:
        for row in matrix:
            for value in row:
                if value is None or not np.isfinite(value):
                    continue
                label = value_formatter(value)
                if len(label) > len(max_label):
                    max_label = label

    for row_idx in range(rows):
        for col_idx in range(cols):
            value = matrix[row_idx][col_idx]
            face = cell_color(value)
            edge = "#d0d0d0"
            patch = FancyBboxPatch(
                (col_idx + gap / 2, row_idx + gap / 2),
                size,
                size,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                linewidth=0.6,
                edgecolor=edge,
                facecolor=face,
            )
            ax.add_patch(patch)
            # Values are rendered after layout to size fonts correctly.

    ax.set_xticks([idx + 0.5 for idx in range(cols)])
    ax.set_yticks([idx + 0.5 for idx in range(rows)])
    tick_fontsize = 8
    max_dim = max(rows, cols)
    if max_dim > 18:
        tick_fontsize = 7
    if max_dim > 24:
        tick_fontsize = 6

    if len(values2) == 1 and str(values2[0]).strip() == "":
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xticklabels(
            [str(value) for value in values2], rotation=45, ha="left", fontsize=tick_fontsize
        )
    ax.set_yticklabels([str(value) for value in values1], fontsize=tick_fontsize)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    if label2:
        ax.set_xlabel(label2)
    ax.set_ylabel(label1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        max_dim = max(rows, cols)
        title_fontsize = 11 if max_dim <= 20 else 10
        wrapped_title = "\n".join(textwrap.wrap(str(title), width=48))
        ax.set_title(wrapped_title, pad=26, fontsize=title_fontsize)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    if show_values and max_label:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        cell_w_px = bbox.width / cols * size
        cell_h_px = bbox.height / rows * size
        from matplotlib.text import Text

        probe = Text(0, 0, max_label, fontsize=value_fontsize, fontweight="bold")
        probe.set_figure(fig)
        text_bbox = probe.get_window_extent(renderer=renderer)
        if text_bbox.width > 0 and text_bbox.height > 0:
            max_by_width = (cell_w_px * 0.9) / text_bbox.width
            max_by_height = (cell_h_px * 0.75) / text_bbox.height
            scale = min(1.0, max_by_width, max_by_height)
            value_fontsize = max(3.0, value_fontsize * scale)

        for row_idx in range(rows):
            for col_idx in range(cols):
                value = matrix[row_idx][col_idx]
                if value is None or not np.isfinite(value):
                    continue
                face = cell_color(value)
                text_color = text_color_for(face, value)
                patch = ax.patches[row_idx * cols + col_idx]
                text = ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    value_formatter(value),
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    fontweight="bold",
                    color=text_color,
                    clip_on=True,
                )
                text.set_clip_path(patch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_stats_chart(result: BacktestResult, output_path: Path) -> Optional[Path]:
    stats = result.stats()
    if not stats:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_return = stats["total_return"] * 100.0
    max_drawdown = stats["max_drawdown"] * 100.0
    win_rate = stats["win_rate"] * 100.0
    profit_factor = stats["profit_factor"]

    labels = ["Total return %", "Max DD %", "Win rate %", "Profit factor"]
    values = [total_return, max_drawdown, win_rate, profit_factor]

    if profit_factor == float("inf"):
        values[-1] = 10.0

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"])
    ax.set_title("Backtest stats")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(
        min(0.0, min(values) * 1.1),
        max(1.0, max(values) * 1.2),
    )

    for bar, label, value in zip(bars, labels, values):
        if label == "Profit factor" and profit_factor == float("inf"):
            text = "inf"
        else:
            text = f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_trade_return_chart(result: BacktestResult, output_path: Path) -> Optional[Path]:
    exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
    if not exit_trades:
        return None

    returns = []
    min_returns = []
    max_returns = []
    for trade in exit_trades:
        returns.append(float(trade.get("return", 0.0)) * 100.0)
        min_returns.append(float(trade.get("min_return", 0.0)) * 100.0)
        max_returns.append(float(trade.get("max_return", 0.0)) * 100.0)

    sorted_rows = sorted(
        zip(returns, min_returns, max_returns),
        key=lambda row: row[0],
    )
    if not sorted_rows:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = list(range(1, len(sorted_rows) + 1))
    sorted_returns = [row[0] for row in sorted_rows]
    sorted_mins = [row[1] for row in sorted_rows]
    sorted_maxs = [row[2] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(x_values, sorted_mins, sorted_maxs, color="#8c8c8c", alpha=0.6, linewidth=1.2)
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in sorted_returns]
    ax.scatter(x_values, sorted_returns, c=colors, s=24, zorder=3)
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_title("Trade returns (sorted) with min/max range")
    ax.set_ylabel("Return %")
    ax.set_xlabel("Trade (sorted by final return)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_trade_distribution_chart(
    result: BacktestResult, output_path: Path
) -> Optional[Path]:
    exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
    if not exit_trades:
        return None

    returns = [float(trade.get("return", 0.0)) * 100.0 for trade in exit_trades]
    if not returns:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(returns)
    bins = int(np.sqrt(count))
    bins = max(5, min(30, bins))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(returns, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")

    mean_val = float(np.mean(returns))
    median_val = float(np.median(returns))
    ax.axvline(mean_val, color="#ff7f0e", linewidth=1.6, label=f"Mean {mean_val:.2f}%")
    ax.axvline(median_val, color="#2ca02c", linewidth=1.6, label=f"Median {median_val:.2f}%")

    ax.set_title("Trade return distribution")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Number of trades")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_holding_time_chart(result: BacktestResult, output_path: Path) -> Optional[Path]:
    exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
    if not exit_trades:
        return None

    durations = []
    numeric_mode = False
    for trade in exit_trades:
        entry_time = trade.get("entry_time")
        exit_time = trade.get("timestamp")
        if entry_time is None or exit_time is None:
            continue

        if isinstance(entry_time, (int, float, np.integer, np.floating)) and isinstance(
            exit_time, (int, float, np.integer, np.floating)
        ):
            duration = float(exit_time) - float(entry_time)
            numeric_mode = True
        else:
            entry_ts = pd.to_datetime(entry_time, errors="coerce")
            exit_ts = pd.to_datetime(exit_time, errors="coerce")
            if pd.isna(entry_ts) or pd.isna(exit_ts):
                continue
            duration = (exit_ts - entry_ts).total_seconds()

        if duration < 0:
            continue
        durations.append(float(duration))

    if not durations:
        return None

    if numeric_mode:
        unit = "steps"
        values = durations
    else:
        median_seconds = float(np.median(durations))
        if median_seconds >= 86400:
            unit = "days"
            divisor = 86400.0
        elif median_seconds >= 3600:
            unit = "hours"
            divisor = 3600.0
        else:
            unit = "minutes"
            divisor = 60.0
        values = [value / divisor for value in durations]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = list(range(1, len(values) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x_values, values, color="#1f77b4", alpha=0.85, width=0.8)
    ax.set_title(f"Holding time per trade ({unit})")
    ax.set_xlabel("Trade")
    ax.set_ylabel(f"Holding time ({unit})")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _format_pct(value: float) -> str:
    value = float(value)
    abs_val = abs(value)
    if abs_val >= 10:
        return f"{value:+.1f}%"
    if abs_val >= 1:
        return f"{value:+.2f}%"
    return f"{value:+.3f}%"


def _build_signed_bins(values: pd.Series) -> Optional[np.ndarray]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return None
    abs_vals = clean.abs()
    max_val = float(abs_vals.max())
    if max_val <= 0:
        return np.array([-1e-6, 0.0, 1e-6])
    quantiles = [0.25, 0.5, 0.75, 0.9]
    q_vals = [float(v) for v in np.quantile(abs_vals, quantiles) if v > 0]
    magnitudes = sorted(set(q_vals + [max_val]))
    negatives = [-v for v in reversed(magnitudes)]
    edges = negatives + [0.0] + magnitudes
    edges = sorted(set(edges))
    if len(edges) < 3:
        mid = max_val * 0.5
        edges = sorted(set([-max_val, -mid, 0.0, mid, max_val]))
    return np.array(edges, dtype=float)


def _build_positive_bins(values: pd.Series) -> Optional[np.ndarray]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return None
    max_val = float(clean.max())
    if max_val <= 0:
        return np.array([0.0, 1e-6, 2e-6], dtype=float)
    quantiles = [0.25, 0.5, 0.75, 0.9]
    q_vals = [float(v) for v in np.quantile(clean, quantiles) if v > 0]
    edges = sorted(set([0.0] + q_vals + [max_val]))
    if len(edges) < 3:
        edges = [0.0, max_val * 0.5, max_val]
    return np.array(sorted(set(edges)), dtype=float)


def _plot_reason_distribution_heatmap(
    frame: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    unit_label: str,
    bin_edges: np.ndarray,
) -> Optional[Path]:
    data = frame.dropna(subset=["exit_reason", value_col]).copy()
    if data.empty:
        return None
    data["bin"] = pd.cut(data[value_col].astype(float), bins=bin_edges, include_lowest=True)
    if data["bin"].isna().all():
        return None
    counts = (
        data.groupby(["exit_reason", "bin"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    if counts.empty:
        return None
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    percents = counts.div(row_sums, axis=0) * 100.0

    bin_labels = []
    for interval in counts.columns:
        left = interval.left
        right = interval.right
        label = f"{_format_pct(left)}..{_format_pct(right)}"
        bin_labels.append(label)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_width = max(10, 1.1 * len(bin_labels))
    fig_height = max(4.5, 0.55 * len(percents))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    matrix = percents.values
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel(unit_label)
    ax.set_ylabel("Exit reason")
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(percents.index)))
    ax.set_yticklabels([str(val) for val in percents.index], fontsize=9)
    ax.grid(False)

    if matrix.size <= 180:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Percent of exits")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_exit_reason_signal_candle_chart(
    frame: pd.DataFrame, output_path: Path
) -> Optional[Path]:
    if frame.empty:
        return None
    required = {"exit_reason", "signal_hour_body_pct"}
    if not required.issubset(frame.columns):
        return None

    bin_edges = _build_positive_bins(frame["signal_hour_body_pct"])
    if bin_edges is None or len(bin_edges) < 3:
        return None

    return _plot_reason_distribution_heatmap(
        frame,
        "signal_hour_body_pct",
        output_path,
        "Signal hour candle body length distribution by exit reason",
        "Signal hour body length bins",
        bin_edges,
    )


def create_exit_reason_holding_time_chart(
    frame: pd.DataFrame, output_path: Path
) -> Optional[Path]:
    if frame.empty:
        return None
    if not {"exit_reason", "entry_time", "exit_time"}.issubset(frame.columns):
        return None

    entries = frame.dropna(subset=["exit_reason", "entry_time", "exit_time"]).copy()
    if entries.empty:
        return None

    entries["entry_time"] = pd.to_datetime(entries["entry_time"], errors="coerce")
    entries["exit_time"] = pd.to_datetime(entries["exit_time"], errors="coerce")
    entries = entries.dropna(subset=["entry_time", "exit_time"])
    if entries.empty:
        return None

    durations = (entries["exit_time"] - entries["entry_time"]).dt.total_seconds()
    entries = entries.assign(duration=durations)
    entries = entries[entries["duration"] >= 0]
    if entries.empty:
        return None

    median_seconds = float(entries["duration"].median())
    if median_seconds >= 86400:
        unit = "days"
        divisor = 86400.0
    elif median_seconds >= 3600:
        unit = "hours"
        divisor = 3600.0
    else:
        unit = "minutes"
        divisor = 60.0

    entries["duration_unit"] = entries["duration"] / divisor
    values = entries["duration_unit"].dropna().astype(float)
    if values.empty:
        return None

    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    edges = sorted(set([float(v) for v in np.quantile(values, quantiles)]))
    if len(edges) < 3:
        max_val = float(values.max())
        edges = [0.0, max_val * 0.5, max_val]
    edges = np.array(edges, dtype=float)
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    if edges[0] > 0:
        edges = np.insert(edges, 0, 0.0)

    return _plot_reason_distribution_heatmap(
        entries,
        "duration_unit",
        output_path,
        "Holding time distribution by exit reason",
        f"Holding time ({unit}) bins",
        edges,
    )


def _resolve_trade_index(data: pd.DataFrame, target: Any) -> Optional[int]:
    if data.empty:
        return None
    if "timestamp" in data.columns:
        ts_col = data["timestamp"]
        if pd.api.types.is_numeric_dtype(ts_col):
            try:
                target_val = float(target)
            except (TypeError, ValueError):
                return None
            diffs = (ts_col.astype(float) - target_val).abs()
            if diffs.isna().all():
                return None
            return int(diffs.idxmin())
        target_ts = pd.to_datetime(target, errors="coerce")
        if pd.isna(target_ts):
            return None
        ts_parsed = pd.to_datetime(ts_col, errors="coerce")
        diffs = (ts_parsed - target_ts).abs()
        if diffs.isna().all():
            return None
        return int(diffs.idxmin())
    if isinstance(target, (int, np.integer)):
        idx = int(target)
        if 0 <= idx < len(data):
            return idx
    return None


def _downsample_ohlc(view: pd.DataFrame, max_candles: int) -> pd.DataFrame:
    if max_candles <= 0 or len(view) <= max_candles:
        return view
    factor = int(np.ceil(len(view) / max_candles))
    if factor <= 1:
        return view
    work = view.reset_index(drop=True).copy()
    work["__bucket"] = np.arange(len(work)) // factor
    agg_map: Dict[str, str] = {}
    if "open" in work.columns:
        agg_map["open"] = "first"
    if "high" in work.columns:
        agg_map["high"] = "max"
    if "low" in work.columns:
        agg_map["low"] = "min"
    if "close" in work.columns:
        agg_map["close"] = "last"
    if "timestamp" in work.columns:
        agg_map["timestamp"] = "last"
    for col in ("ema_fast", "ema_slow", "macro_ema"):
        if col in work.columns:
            agg_map[col] = "last"
    reduced = work.groupby("__bucket", as_index=False).agg(agg_map)
    return reduced.drop(columns="__bucket")


def create_trade_snapshot_chart(
    data: pd.DataFrame,
    trade: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = None,
    lookback: int = 30,
    lookforward: int = 10,
    max_candles: int = 240,
) -> Optional[Path]:
    if data.empty:
        return None
    entry_time = trade.get("entry_time")
    exit_time = trade.get("timestamp")
    entry_idx = _resolve_trade_index(data, entry_time)
    exit_idx = _resolve_trade_index(data, exit_time)
    if entry_idx is None or exit_idx is None:
        return None

    start_idx = max(0, min(entry_idx, exit_idx) - lookback)
    end_idx = min(len(data) - 1, max(entry_idx, exit_idx) + lookforward)
    view = data.iloc[start_idx : end_idx + 1].copy()
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(view.columns):
        return None

    entry_time = trade.get("entry_time")
    exit_time = trade.get("timestamp")
    if len(view) > max_candles:
        view = _downsample_ohlc(view, max_candles=max_candles)
        entry_rel = _resolve_trade_index(view, entry_time)
        exit_rel = _resolve_trade_index(view, exit_time)
        if entry_rel is None or exit_rel is None:
            return None
    else:
        entry_rel = entry_idx - start_idx
        exit_rel = exit_idx - start_idx

    x_values = np.arange(len(view))
    timestamps = None
    if "timestamp" in view.columns:
        ts = pd.to_datetime(view["timestamp"], errors="coerce")
        if not ts.isna().all():
            timestamps = ts

    entry_price = trade.get("entry_price")
    if entry_price is None:
        entry_row = data.iloc[entry_idx]
        entry_price = float(entry_row.get("open", entry_row.get("close", np.nan)))
    exit_price = trade.get("price")
    if exit_price is None:
        exit_row = data.iloc[exit_idx]
        exit_price = float(exit_row.get("open", exit_row.get("close", np.nan)))

    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        return number

    tp_price = _coerce_float(
        trade.get("rr_tp_price", trade.get("tp_price", trade.get("tp")))
    )
    sl_price = _coerce_float(
        trade.get(
            "rr_stop_price",
            trade.get("stop_price", trade.get("sl_price", trade.get("sl"))),
        )
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    from matplotlib.patches import Rectangle

    up_color = "#2ca02c"
    down_color = "#d62728"
    neutral_color = "#7f7f7f"
    candle_width = 0.6
    wick_width = 1.0

    ohlc = view[["open", "high", "low", "close"]].astype(float)
    for i, (open_p, high_p, low_p, close_p) in enumerate(ohlc.itertuples(index=False)):
        if np.isnan(open_p) or np.isnan(high_p) or np.isnan(low_p) or np.isnan(close_p):
            continue
        if close_p > open_p:
            color = up_color
        elif close_p < open_p:
            color = down_color
        else:
            color = neutral_color

        ax.vlines(
            x_values[i],
            low_p,
            high_p,
            color=color,
            linewidth=wick_width,
            zorder=2,
        )
        body_low = min(open_p, close_p)
        body_height = abs(close_p - open_p)
        if body_height == 0:
            ax.hlines(
                open_p,
                x_values[i] - candle_width / 2,
                x_values[i] + candle_width / 2,
                color=color,
                linewidth=2.0,
                zorder=3,
            )
        else:
            rect = Rectangle(
                (x_values[i] - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                zorder=3,
            )
            ax.add_patch(rect)
    ax.scatter(
        [x_values[entry_rel]],
        [entry_price],
        color=up_color,
        s=48,
        zorder=5,
        label="Entry",
    )
    ax.scatter(
        [x_values[exit_rel]],
        [exit_price],
        color=down_color,
        s=48,
        zorder=5,
        label="Exit",
    )
    ax.axvline(x_values[entry_rel], color=up_color, linewidth=0.8, alpha=0.4)
    ax.axvline(x_values[exit_rel], color=down_color, linewidth=0.8, alpha=0.4)
    if tp_price is not None:
        ax.axhline(
            tp_price,
            color=up_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="TP",
        )
    if sl_price is not None:
        ax.axhline(
            sl_price,
            color=down_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="SL",
        )
    if "ema_fast" in view.columns:
        ax.plot(
            x_values,
            view["ema_fast"].astype(float),
            color="#ff7f0e",
            linewidth=1.1,
            label="EMA fast",
        )
    if "ema_slow" in view.columns:
        ax.plot(
            x_values,
            view["ema_slow"].astype(float),
            color="#9467bd",
            linewidth=1.1,
            label="EMA slow",
        )
    if "macro_ema" in view.columns:
        ax.plot(
            x_values,
            view["macro_ema"].astype(float),
            color="#8c564b",
            linewidth=1.1,
            label="Macro EMA",
        )
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    if timestamps is not None:
        tick_count = min(6, len(view))
        tick_idx = np.linspace(0, len(view) - 1, num=tick_count, dtype=int)
        ax.set_xticks(x_values[tick_idx])
        labels = []
        for idx in tick_idx:
            ts = timestamps.iloc[idx]
            if pd.isna(ts):
                labels.append("")
            else:
                labels.append(ts.strftime("%Y-%m-%d %H:%M"))
        ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_xticks([])

    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_trade_pnl_curve(
    data: pd.DataFrame,
    trade: Dict[str, Any],
    output_path: Path,
    max_candles: int = 240,
) -> Optional[Path]:
    if data.empty:
        return None
    entry_time = trade.get("entry_time")
    exit_time = trade.get("timestamp")
    entry_idx = _resolve_trade_index(data, entry_time)
    exit_idx = _resolve_trade_index(data, exit_time)
    if entry_idx is None or exit_idx is None:
        return None
    start_idx = min(entry_idx, exit_idx)
    end_idx = max(entry_idx, exit_idx)
    view = data.iloc[start_idx : end_idx + 1].copy()
    if view.empty or "close" not in view.columns:
        return None
    if len(view) > max_candles:
        view = _downsample_ohlc(view, max_candles=max_candles)
    entry_price = trade.get("entry_price")
    if entry_price is None:
        entry_row = data.iloc[entry_idx]
        entry_price = float(entry_row.get("open", entry_row.get("close", np.nan)))
    if entry_price is None or entry_price == 0:
        return None
    direction = int(trade.get("direction", 0))
    if direction == 0:
        return None
    returns = (view["close"].astype(float) - float(entry_price)) / float(entry_price)
    returns = returns * (1.0 if direction > 0 else -1.0) * 100.0

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(np.arange(len(returns)), returns, color="#1f77b4", linewidth=1.6)
    ax.axhline(0, color="#7f7f7f", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Return (%)")
    ax.set_title("Trade return curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_daily_return_heatmaps(
    result: BacktestResult, output_path: Path
) -> list[Path]:
    if result.equity_curve.empty:
        return []
    if "timestamp" not in result.equity_curve.columns:
        return []

    df = result.equity_curve.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return []

    df = df.sort_values("timestamp").set_index("timestamp")
    daily_equity = df["equity"].astype(float).resample("1D").last()
    daily_equity.index = pd.to_datetime(daily_equity.index, errors="coerce")
    if daily_equity.index.tz is not None:
        daily_equity.index = daily_equity.index.tz_convert(None)
    daily_equity.index = daily_equity.index.normalize()
    daily_returns = daily_equity.pct_change().dropna()
    if daily_returns.empty:
        return []

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import calendar
    from matplotlib.patches import FancyBboxPatch

    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    output_paths: list[Path] = []

    start_date = daily_returns.index.min().date()
    end_date = daily_returns.index.max().date()

    daily_return_map = {
        ts.date(): float(value) * 100.0 for ts, value in daily_returns.items()
    }

    current_month = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    cal = calendar.Calendar(firstweekday=0)

    months_data: list[dict[str, object]] = []
    while current_month <= end_month:
        year = current_month.year
        month = current_month.month
        month_weeks = cal.monthdatescalendar(year, month)
        weeks = len(month_weeks)

        matrix = np.full((weeks, 7), np.nan, dtype=float)
        in_range = np.zeros((weeks, 7), dtype=bool)
        for week_index, week in enumerate(month_weeks):
            for weekday, day in enumerate(week):
                if day.month != month:
                    continue
                if day < start_date or day > end_date:
                    continue
                in_range[week_index, weekday] = True
                value = daily_return_map.get(day)
                if value is None:
                    continue
                matrix[week_index, weekday] = value

        months_data.append(
            {
                "year": year,
                "month": month,
                "weeks": weeks,
                "month_weeks": month_weeks,
                "matrix": matrix,
                "in_range": in_range,
            }
        )

        if month == 12:
            current_month = date(year + 1, 1, 1)
        else:
            current_month = date(year, month + 1, 1)

    if not months_data:
        return []

    def cell_color(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "#f2f2f2"
        if value < 0.0:
            return "#ef5350"
        if value > 0.0:
            return "#26a69a"
        return "#dcdcdc"

    chunk_size = 3
    for chunk_start in range(0, len(months_data), chunk_size):
        chunk = months_data[chunk_start : chunk_start + chunk_size]
        months_in_chunk = len(chunk)
        max_weeks = max(int(item["weeks"]) for item in chunk)

        cell_size = 0.55
        ax_width = 7 * cell_size
        ax_height = max_weeks * cell_size
        fig_width = months_in_chunk * ax_width + 1.4
        fig_height = ax_height + 1.2

        fig, axes = plt.subplots(
            1, months_in_chunk, figsize=(fig_width, fig_height), squeeze=False
        )

        for ax_index, item in enumerate(chunk):
            ax = axes[0, ax_index]
            year = int(item["year"])
            month = int(item["month"])
            weeks = int(item["weeks"])
            month_weeks = item["month_weeks"]
            matrix = item["matrix"]
            in_range = item["in_range"]

            ax.set_facecolor("#f7f7f7")

            for week_index, week in enumerate(month_weeks):
                for weekday, day in enumerate(week):
                    if day.month != month:
                        continue
                    if not in_range[week_index, weekday]:
                        continue
                    value = matrix[week_index, weekday]
                    face = cell_color(value)
                    patch = FancyBboxPatch(
                        (weekday - 0.5 + 0.05, week_index - 0.5 + 0.05),
                        0.9,
                        0.9,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=0.6,
                        edgecolor="#d0d0d0",
                        facecolor=face,
                    )
                    ax.add_patch(patch)

            for week_index, week in enumerate(month_weeks):
                for weekday, day in enumerate(week):
                    if day.month != month:
                        continue
                    if not in_range[week_index, weekday]:
                        continue
                    value = matrix[week_index, weekday]
                    if np.isfinite(value):
                        text_color = "white" if value != 0.0 else "#333333"
                        ax.text(
                            weekday,
                            week_index - 0.12,
                            f"{day.day}",
                            ha="center",
                            va="center",
                            fontsize=8.5,
                            fontweight="bold",
                            color=text_color,
                        )
                        ax.text(
                            weekday,
                            week_index + 0.2,
                            f"{value:+.2f}%",
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color=text_color,
                        )
                    else:
                        ax.text(
                            weekday,
                            week_index,
                            f"{day.day}",
                            ha="center",
                            va="center",
                            fontsize=8.5,
                            fontweight="bold",
                            color="#666666",
                        )

            ax.set_xticks(range(7))
            ax.set_xticklabels(weekday_labels, fontsize=8)
            ax.xaxis.tick_top()
            ax.set_yticks([])
            ax.set_title(f"{year}-{month:02d}", fontsize=11, pad=24)
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(weeks - 0.5, -0.5)
            ax.set_aspect("equal", adjustable="box")

            ax.grid(which="major", visible=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_start_label = chunk[0]
        chunk_end_label = chunk[-1]
        output_file = output_path.with_name(
            (
                f"{output_path.stem}_"
                f"{int(chunk_start_label['year'])}{int(chunk_start_label['month']):02d}-"
                f"{int(chunk_end_label['year'])}{int(chunk_end_label['month']):02d}"
                f"{output_path.suffix}"
            )
        )
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
        output_paths.append(output_file)

    return output_paths
