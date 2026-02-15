from __future__ import annotations

from pathlib import Path

from backtest.config import load_settings
from backtest.discord_notifier import DiscordNotifier
from backtest.runner import run_backtest_from_settings


def _print_progress(done: int, total: int) -> None:
    if total <= 0:
        pct = 100
    else:
        pct = int((done / total) * 100)
    print(f"\rBacktest progress: {pct}%", end="", flush=True)
    if done >= total:
        print()


def main() -> None:
    settings = load_settings(Path("config/settings.yaml"))

    message, _, chart_paths = run_backtest_from_settings(settings, progress_cb=_print_progress)

    discord_cfg = settings.discord
    if discord_cfg.get("enabled"):
        notifier = DiscordNotifier(
            bot_token=discord_cfg.get("bot_token", ""),
            channel_id=discord_cfg.get("channel_id", ""),
        )
        notifier.send_message(message, chart_paths=chart_paths)
    else:
        print(message)
        if chart_paths:
            for path in chart_paths:
                print(f"Output saved to: {path}")


if __name__ == "__main__":
    main()
