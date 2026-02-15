from __future__ import annotations

import logging
from pathlib import Path

from backtest.config import load_settings
from backtest.discord_bot import BacktestBot


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[{asctime}] [{levelname:<8}] {name}: {message}",
        style="{",
    )
    settings_path = Path("config/settings.yaml")
    settings = load_settings(settings_path)
    discord_cfg = settings.discord
    if not discord_cfg.get("enabled"):
        raise SystemExit("discord.enabled must be true in config/settings.yaml")
    bot_token = discord_cfg.get("bot_token")
    if not bot_token or "PUT_DISCORD_BOT_TOKEN_HERE" in bot_token:
        raise SystemExit("Set discord.bot_token in config/settings.yaml")

    bot = BacktestBot(settings, settings_path)
    bot.run(bot_token)


if __name__ == "__main__":
    main()
