from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[Dict[str, Any]]
    add_buy_debug: Dict[str, Any] = field(default_factory=dict)
    final_cash: float = 0.0
    open_positions: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        start = self.equity_curve["equity"].iloc[0]
        end = self.equity_curve["equity"].iloc[-1]
        if start == 0:
            return 0.0
        return (end - start) / start

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        equity = self.equity_curve["equity"].astype(float)
        peak = equity.cummax()
        drawdown = (equity / peak) - 1.0
        return float(drawdown.min())

    def _exit_trades(self) -> List[Dict[str, float]]:
        return [trade for trade in self.trades if trade.get("type") == "exit"]

    def _partial_exit_trades(self) -> List[Dict[str, float]]:
        return [trade for trade in self.trades if trade.get("type") == "exit_partial"]

    def _trade_pnls(self) -> List[float]:
        return [float(trade.get("pnl", 0.0)) for trade in self._exit_trades()]

    def _add_buy_trade_stats(self) -> Dict[str, Any]:
        current: Dict[int, Dict[str, Any] | None] = {1: None, -1: None}
        trade_summaries: List[Dict[str, Any]] = []

        for trade in self.trades:
            ttype = trade.get("type")
            direction = int(trade.get("direction", 0))
            if ttype == "entry":
                if direction in current:
                    current[direction] = {"add_count": 0}
            elif ttype == "add":
                state = current.get(direction)
                if state is not None:
                    state["add_count"] = int(state.get("add_count", 0)) + 1
            elif ttype == "exit":
                state = current.get(direction)
                add_count = int(state.get("add_count", 0)) if state else 0
                pnl = float(trade.get("pnl", 0.0))
                exit_reason = trade.get("exit_reason") or "unknown"
                trade_summaries.append(
                    {
                        "add_count": add_count,
                        "pnl": pnl,
                        "win": pnl > 0,
                        "exit_reason": exit_reason,
                    }
                )
                if direction in current:
                    current[direction] = None

        total_trades = len(trade_summaries)
        add_trade_count = sum(1 for item in trade_summaries if item["add_count"] > 0)
        add_ratio = add_trade_count / total_trades if total_trades > 0 else 0.0
        max_adds = max((item["add_count"] for item in trade_summaries), default=0)
        avg_add_all = (
            sum(item["add_count"] for item in trade_summaries) / total_trades
            if total_trades > 0
            else 0.0
        )
        avg_add_with = (
            sum(item["add_count"] for item in trade_summaries if item["add_count"] > 0)
            / add_trade_count
            if add_trade_count > 0
            else 0.0
        )

        count_distribution: Dict[int, int] = {}
        win_rate_by_times: Dict[int, Dict[str, float]] = {}
        exit_reasons_by_times: Dict[int, Dict[str, int]] = {}
        for item in trade_summaries:
            count = int(item["add_count"])
            count_distribution[count] = count_distribution.get(count, 0) + 1

            bucket = win_rate_by_times.get(count)
            if bucket is None:
                bucket = {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0}
                win_rate_by_times[count] = bucket
            bucket["total"] += 1
            if item["win"]:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
            bucket["win_rate"] = (
                float(bucket["wins"]) / float(bucket["total"]) if bucket["total"] > 0 else 0.0
            )

            reason_counts = exit_reasons_by_times.get(count)
            if reason_counts is None:
                reason_counts = {}
                exit_reasons_by_times[count] = reason_counts
            reason = str(item["exit_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            "add_buy_trade_count": add_trade_count,
            "add_buy_trade_ratio": add_ratio,
            "add_buy_avg_times_all": avg_add_all,
            "add_buy_avg_times_with_add": avg_add_with,
            "add_buy_max_times": max_adds,
            "add_buy_count_distribution": count_distribution,
            "add_buy_win_rate_by_times": win_rate_by_times,
            "add_buy_exit_reason_counts_by_times": exit_reasons_by_times,
        }

    def stats(self) -> Dict[str, Any]:
        exit_trades = self._exit_trades()
        partial_exit_trades = self._partial_exit_trades()

        def _count_reasons(trades: List[Dict[str, Any]]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for trade in trades:
                reason = trade.get("exit_reason") or "unknown"
                counts[reason] = counts.get(reason, 0) + 1
            return counts

        exit_reason_counts = _count_reasons(exit_trades + partial_exit_trades)
        exit_reason_counts_full = _count_reasons(exit_trades)
        exit_reason_counts_partial = _count_reasons(partial_exit_trades)
        pnls = [float(trade.get("pnl", 0.0)) for trade in exit_trades]
        returns = [float(trade.get("return", 0.0)) for trade in exit_trades]
        min_returns = [float(trade.get("min_return", 0.0)) for trade in exit_trades]
        max_returns = [float(trade.get("max_return", 0.0)) for trade in exit_trades]
        total_trades = len(exit_trades)
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        long_returns = [
            ret for trade, ret in zip(exit_trades, returns) if trade.get("direction") == 1
        ]
        short_returns = [
            ret for trade, ret in zip(exit_trades, returns) if trade.get("direction") == -1
        ]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        avg_trade = sum(pnls) / total_trades if total_trades > 0 else 0.0
        avg_win = sum(wins) / win_count if win_count > 0 else 0.0
        avg_loss = sum(losses) / loss_count if loss_count > 0 else 0.0
        avg_return = sum(returns) / total_trades if total_trades > 0 else 0.0
        avg_min_return = sum(min_returns) / total_trades if total_trades > 0 else 0.0
        avg_max_return = sum(max_returns) / total_trades if total_trades > 0 else 0.0

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0
        for pnl in pnls:
            if pnl > 0:
                current_win += 1
                current_loss = 0
            elif pnl < 0:
                current_loss += 1
                current_win = 0
            else:
                current_win = 0
                current_loss = 0
            if current_win > max_win_streak:
                max_win_streak = current_win
            if current_loss > max_loss_streak:
                max_loss_streak = current_loss

        long_count = len(long_returns)
        short_count = len(short_returns)
        long_ratio = long_count / total_trades if total_trades > 0 else 0.0
        short_ratio = short_count / total_trades if total_trades > 0 else 0.0
        long_avg_return = sum(long_returns) / long_count if long_count > 0 else 0.0
        short_avg_return = sum(short_returns) / short_count if short_count > 0 else 0.0

        start_equity = float(self.equity_curve["equity"].iloc[0]) if not self.equity_curve.empty else 0.0
        final_equity = float(self.equity_curve["equity"].iloc[-1]) if not self.equity_curve.empty else 0.0
        add_buy_stats = self._add_buy_trade_stats()

        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "total_trades": total_trades,
            "num_wins": win_count,
            "num_losses": loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_return": avg_return,
            "avg_min_return": avg_min_return,
            "avg_max_return": avg_max_return,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "long_trades": long_count,
            "short_trades": short_count,
            "long_ratio": long_ratio,
            "short_ratio": short_ratio,
            "long_avg_return": long_avg_return,
            "short_avg_return": short_avg_return,
            "start_equity": start_equity,
            "final_equity": final_equity,
            "exit_reason_counts": exit_reason_counts,
            "exit_reason_counts_full": exit_reason_counts_full,
            "exit_reason_counts_partial": exit_reason_counts_partial,
            "exit_reason_total_full": len(exit_trades),
            "exit_reason_total_partial": len(partial_exit_trades),
            "add_buy_debug": self.add_buy_debug,
            **add_buy_stats,
        }
