from __future__ import annotations

import asyncio
import copy
import csv
import json
import logging
import random
import re
import shlex
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Any, Dict, List, Sequence, Tuple, Callable, Union, Set

import discord
import numpy as np
import pandas as pd
import yaml
from discord import app_commands
from discord.ext import commands
from discord.ui import View, Select, Button, Modal, TextInput
from zoneinfo import ZoneInfo

from backtest.binance import BinanceAPIError, BinanceFuturesClient, BinanceSpotClient
from backtest.config import Settings, save_settings, load_settings
from backtest.analysis import EntryDiffConfig
from backtest.runner import (
    _normalize_cheatkey_params,
    create_settings_snapshot,
    run_backtest_with_entry_diff_result,
    run_backtest_with_result,
    run_backtest_summary,
)
from backtest.data import download_price_data
from backtest.realtime import RealtimeRunner, RunnerPayload
from backtest.strategy_loader import list_strategy_names
from backtest.plot import (
    create_costom_loss_chart,
    create_equity_overlay_monthly_best_chart,
    create_equity_mdd_monthly_stats_chart,
    create_equity_price_chart,
    create_monthly_return_stability_chart,
    create_equity_price_mdd_chart,
    create_entry_minute_win_rate_chart,
    create_entry_minute_tp_rate_chart,
    create_grid_heatmap,
    create_trade_pnl_curve,
    create_trade_snapshot_chart,
    compute_entry_minute_win_rate,
    compute_entry_minute_tp_rate,
    compute_monthly_return_stats,
)

logger = logging.getLogger(__name__)


class GridProgressHandle:
    def __init__(
        self,
        run_id: str,
        channel_id: int,
        total: int,
        message: Optional[discord.Message] = None,
    ) -> None:
        self.run_id = run_id
        self.channel_id = channel_id
        self.total = total
        self.done = 0
        self.message = message


class PendingGridOutput:
    def __init__(
        self,
        channel_id: int,
        content: str,
        image_paths: Optional[Sequence[Path]] = None,
        data_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        self.channel_id = channel_id
        self.content = content
        self.image_paths = list(image_paths) if image_paths else []
        self.data_paths = list(data_paths) if data_paths else []


@dataclass
class PositionStatus:
    side: str
    active: bool
    notional: float = 0.0
    qty: float = 0.0
    leverage: float = 1.0
    tp_pnl: Optional[float] = None
    sl_pnl: Optional[float] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    pnl_pct: float = 0.0
    pnl_amount: float = 0.0


@dataclass
class StatusSnapshot:
    symbol: str
    interval: str
    price: float
    equity: float
    free_balance: float
    long: PositionStatus
    short: PositionStatus
    updated_at: float


@dataclass
class WordleGame:
    word: str
    guesses: List[str]
    max_attempts: int = 6
    completed: bool = False

    @property
    def attempts_left(self) -> int:
        return max(self.max_attempts - len(self.guesses), 0)


@dataclass
class LiveOrderDraft:
    symbol: str
    side: str
    position_side: str
    leverage: float
    usdt_amount: float
    price: float
    qty: float
    reduce_only: bool
    maker_offset_bps: float


@dataclass
class ClosePositionDraft:
    symbol: str
    side: str
    position_side: str
    qty: float
    price: float
    maker_offset_bps: float


WORDLE_WORDS = (
    "about", "above", "actor", "acute", "admit", "adopt", "adult", "after", "again", "agent",
    "agree", "ahead", "alarm", "album", "alert", "alien", "alive", "allow", "alone", "along",
    "alter", "among", "anger", "angle", "angry", "apple", "apply", "arena", "argue", "arise",
    "array", "aside", "asset", "audio", "audit", "avoid", "award", "aware", "badly", "baker",
    "basic", "beach", "began", "begin", "being", "below", "bench", "birth", "black", "blame",
    "blend", "blind", "block", "blood", "board", "brain", "brand", "bread", "break", "brick",
    "brief", "bring", "broad", "brown", "build", "buyer", "cable", "carry", "catch", "cause",
    "chain", "chair", "chart", "chase", "cheap", "check", "chess", "chief", "child", "china",
    "choose", "claim", "class", "clean", "clear", "click", "clock", "close", "coach", "coast",
    "color", "could", "count", "court", "cover", "craft", "crash", "cream", "crime", "cross",
    "crowd", "crown", "daily", "dance", "death", "delay", "depth", "devil", "dirty", "doubt",
    "dozen", "draft", "drama", "dream", "dress", "drink", "drive", "drove", "dying", "eager",
    "early", "earth", "eight", "elite", "empty", "enemy", "enjoy", "enter", "equal", "error",
    "event", "every", "exact", "exist", "extra", "faith", "false", "fault", "fiber", "field",
    "fifth", "fifty", "fight", "final", "first", "fixed", "flash", "fleet", "floor", "fluid",
    "focus", "force", "forth", "forty", "forum", "found", "frame", "fresh", "front", "fruit",
    "fully", "funny", "giant", "given", "glass", "globe", "going", "grace", "grade", "grand",
    "grant", "grass", "great", "green", "gross", "group", "grown", "guard", "guess", "guest",
    "guide", "happy", "harsh", "heart", "heavy", "hence", "henry", "horse", "hotel", "house",
    "human", "ideal", "image", "index", "inner", "input", "issue", "joint", "judge", "known",
    "label", "large", "laser", "later", "laugh", "layer", "learn", "least", "leave", "legal",
    "level", "light", "limit", "local", "logic", "loose", "lower", "lucky", "lunch", "major",
    "maker", "march", "match", "maybe", "media", "metal", "might", "minor", "model", "money",
    "month", "moral", "motor", "mount", "mouse", "mouth", "movie", "music", "needs", "never",
    "newly", "night", "noise", "north", "novel", "nurse", "occur", "ocean", "offer", "often",
    "order", "other", "owner", "panel", "paper", "party", "peace", "peter", "phase", "phone",
    "piece", "pilot", "pitch", "place", "plain", "plane", "plant", "plate", "point", "power",
    "press", "price", "pride", "prime", "print", "prior", "prove", "queen", "quick", "quiet",
    "quite", "radio", "raise", "range", "rapid", "ratio", "reach", "ready", "refer", "right",
    "river", "rough", "round", "route", "royal", "rural", "scale", "scene", "scope", "score",
    "sense", "serve", "seven", "shall", "shape", "share", "sharp", "sheet", "shelf", "shell",
    "shift", "shirt", "shock", "shoot", "short", "shown", "sight", "since", "skill", "sleep",
    "small", "smart", "smile", "smith", "smoke", "solid", "solve", "sorry", "sound", "south",
    "space", "spare", "speak", "speed", "spend", "spent", "split", "sport", "staff", "stage",
    "stake", "stand", "start", "state", "steam", "steel", "stick", "still", "stock", "stone",
    "stood", "store", "storm", "story", "strip", "stuck", "study", "stuff", "style", "sugar",
    "suite", "super", "table", "taken", "taste", "teach", "teeth", "terry", "thank", "their",
    "theme", "there", "these", "thick", "thing", "think", "third", "those", "three", "threw",
    "throw", "tight", "times", "tired", "title", "today", "topic", "total", "touch", "tough",
    "tower", "track", "trade", "train", "treat", "trend", "trial", "tried", "tries", "truck",
    "truly", "trust", "truth", "twice", "under", "union", "unity", "until", "upper", "upset",
    "urban", "usage", "usual", "valid", "value", "video", "visit", "vital", "voice", "waste",
    "watch", "water", "wheel", "where", "which", "while", "white", "whole", "whose", "woman",
    "women", "world", "worry", "worse", "worst", "worth", "would", "wound", "write", "wrong",
    "wrote", "yield", "young", "youth",
)


class ConfigValueModal(Modal):
    def __init__(
        self,
        editor: "ConfigEditorView",
        full_key: str,
        current_value: Any,
    ) -> None:
        title = f"Edit {full_key}"
        if len(title) > 45:
            title = title[:42] + "..."
        super().__init__(title=title)
        self.editor = editor
        self.full_key = full_key
        self.value_input = TextInput(
            label="New value",
            default=str(current_value),
            required=True,
            max_length=200,
        )
        self.add_item(self.value_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        updated = self.editor.apply_change(self.full_key, self.value_input.value)
        if not updated:
            await interaction.response.send_message(
                "Failed to update value. Please try again.", ephemeral=True
            )
            return
        await self.editor.refresh_message(interaction)
        await interaction.followup.send(
            f"Updated {self.full_key}.", ephemeral=True
        )


class DownloadDataModal(Modal):
    def __init__(self, bot: "BacktestBot") -> None:
        super().__init__(title="Download Data")
        self.bot = bot
        defaults = bot._get_download_defaults()
        self.symbol = TextInput(
            label="Symbol (e.g. BTCUSDT)",
            default=defaults.get("symbol", ""),
            required=True,
            max_length=20,
        )
        self.interval = TextInput(
            label="Interval (e.g. 1m, 5m, 1h)",
            default=defaults.get("interval", ""),
            required=True,
            max_length=10,
        )
        self.start = TextInput(
            label="Start datetime (YYYY-MM-DD HH:MM:SS)",
            default=defaults.get("start", ""),
            required=True,
            max_length=19,
        )
        self.end = TextInput(
            label="End datetime (YYYY-MM-DD HH:MM:SS)",
            default=defaults.get("end", ""),
            required=True,
            max_length=19,
        )
        self.market = TextInput(
            label="Market (futures or spot)",
            default=defaults.get("market", "futures"),
            required=True,
            max_length=10,
        )

        self.add_item(self.symbol)
        self.add_item(self.interval)
        self.add_item(self.start)
        self.add_item(self.end)
        self.add_item(self.market)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.bot._handle_download_request(
            interaction,
            self.symbol.value.strip(),
            self.interval.value.strip(),
            self.start.value.strip(),
            self.end.value.strip(),
            self.market.value.strip(),
        )


class DownloadDataButtonView(View):
    def __init__(self, bot: "BacktestBot") -> None:
        super().__init__(timeout=300)
        self.bot = bot

    @discord.ui.button(label="Open Download Form", style=discord.ButtonStyle.primary)
    async def open_form(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        await interaction.response.send_modal(DownloadDataModal(self.bot))


class SettingSaveModal(Modal):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(title="Save Current Settings")
        self.bot = bot
        self.requester_id = requester_id
        self.name_input = TextInput(
            label="Profile name",
            required=True,
            max_length=50,
        )
        self.description_input = TextInput(
            label="Description (optional)",
            required=False,
            max_length=200,
        )
        self.add_item(self.name_input)
        self.add_item(self.description_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the save can submit.",
                ephemeral=True,
            )
            return
        name = self.name_input.value.strip()
        description = self.description_input.value.strip()
        path, err = self.bot._save_setting_profile(
            name,
            description=description or None,
            overwrite=False,
        )
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        await interaction.response.send_message(
            f"Saved current settings to `{path.name}`.",
            ephemeral=True,
        )


class SettingOverwriteSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        profiles: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.requester_id = requester_id
        options = [
            discord.SelectOption(label=name[:100], value=name) for name in profiles
        ]
        super().__init__(
            placeholder="Select a profile to overwrite",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the save can overwrite.",
                ephemeral=True,
            )
            return
        name = self.values[0]
        path, err = self.bot._save_setting_profile(
            name,
            description=None,
            overwrite=True,
        )
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        await interaction.response.send_message(
            f"Overwrote `{path.name}` with current settings.",
            ephemeral=True,
        )


class SettingApplySelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        profiles: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.requester_id = requester_id
        options: List[discord.SelectOption] = []
        for name in profiles:
            label = bot._format_setting_profile_label(name)
            options.append(discord.SelectOption(label=label[:100], value=name))
        super().__init__(
            placeholder="Select a profile to apply",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested apply can use this.",
                ephemeral=True,
            )
            return
        name = self.values[0]
        ok, message = await self.bot._apply_setting_profile(name)
        await interaction.response.send_message(
            message,
            ephemeral=not ok,
        )


class LiveOrderModal(Modal):
    def __init__(
        self,
        bot: "BacktestBot",
        requester_id: int,
        default_symbol: str,
        default_leverage: float,
    ) -> None:
        super().__init__(title="Place Live Futures Order")
        self.bot = bot
        self.requester_id = requester_id
        self.default_symbol = default_symbol
        self.default_leverage = default_leverage
        self.side = TextInput(
            label="Side",
            placeholder="long / short (or buy / sell)",
            required=True,
            max_length=10,
        )
        self.usdt_amount = TextInput(
            label="Amount (USDT)",
            placeholder="e.g. 10",
            required=True,
            max_length=20,
        )
        self.price = TextInput(
            label="Price (optional)",
            placeholder="Leave blank to use current price",
            required=False,
            max_length=20,
        )
        self.symbol = TextInput(
            label=f"Symbol (default {default_symbol})",
            placeholder=default_symbol,
            required=False,
            max_length=20,
        )
        self.leverage = TextInput(
            label=f"Leverage (default {default_leverage})",
            placeholder=str(default_leverage),
            required=False,
            max_length=10,
        )
        for item in (
            self.side,
            self.usdt_amount,
            self.price,
            self.symbol,
            self.leverage,
        ):
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can submit this order.",
                ephemeral=True,
            )
            return
        modal_fields = {
            "side": self.side.value,
            "usdt_amount": self.usdt_amount.value,
            "price": self.price.value,
            "symbol": self.symbol.value,
            "leverage": self.leverage.value,
        }
        await self.bot._handle_live_order(interaction, "", modal_fields=modal_fields)


class SettingOverwriteView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        profiles: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.message: Optional[discord.Message] = None
        self.add_item(SettingOverwriteSelect(bot, profiles, requester_id))

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)


class SettingApplyView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        profiles: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.message: Optional[discord.Message] = None
        self.add_item(SettingApplySelect(bot, profiles, requester_id))

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)


class LiveOrderStartView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        requester_id: int,
        default_symbol: str,
        default_leverage: float,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.default_symbol = default_symbol
        self.default_leverage = default_leverage
        self.message: Optional[discord.Message] = None

    @discord.ui.button(label="Open Order Form", style=discord.ButtonStyle.primary)
    async def open_form(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can open this form.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(
            LiveOrderModal(
                self.bot,
                self.requester_id,
                self.default_symbol,
                self.default_leverage,
            )
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can cancel this.",
                ephemeral=True,
            )
            return
        for item in self.children:
            item.disabled = True
        if interaction.response.is_done():
            await interaction.followup.send("Cancelled.", ephemeral=True)
        else:
            await interaction.response.send_message("Cancelled.", ephemeral=True)
        if self.message:
            await self.message.edit(view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)


class LiveOrderConfirmView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        requester_id: int,
        draft: LiveOrderDraft,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.draft = draft
        self.message: Optional[discord.Message] = None

    @discord.ui.button(label="Place Order", style=discord.ButtonStyle.primary)
    async def place_order(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can place this order.",
                ephemeral=True,
            )
            return
        if not interaction.response.is_done():
            try:
                await interaction.response.defer(ephemeral=True, thinking=True)
            except (discord.NotFound, discord.errors.NotFound, discord.HTTPException):
                pass
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)
        await self.bot._execute_live_order(interaction, self.draft)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can cancel this.",
                ephemeral=True,
            )
            return
        for item in self.children:
            item.disabled = True
        if interaction.response.is_done():
            await interaction.followup.send("Cancelled.", ephemeral=True)
        else:
            await interaction.response.send_message("Cancelled.", ephemeral=True)
        if self.message:
            await self.message.edit(view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)


class ClosePositionSelect(discord.ui.Select):
    def __init__(
        self,
        bot: "BacktestBot",
        drafts: List[ClosePositionDraft],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.drafts = drafts
        self.requester_id = requester_id
        options: List[discord.SelectOption] = []
        for idx, draft in enumerate(drafts):
            label = f"{draft.symbol} {draft.position_side}"
            desc = f"qty {draft.qty:.6g} @ {bot._fmt_price(draft.price)}"
            options.append(
                discord.SelectOption(
                    label=label[:100],
                    value=str(idx),
                    description=desc[:100],
                )
            )
        super().__init__(
            placeholder="Select a position to close",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can close a position.",
                ephemeral=True,
            )
            return
        idx = int(self.values[0])
        draft = self.drafts[idx]
        view = ClosePositionConfirmView(self.bot, self.requester_id, draft)
        text = (
            "Confirm close:\n"
            f"- Symbol: {draft.symbol}\n"
            f"- Side: {draft.position_side}\n"
            f"- Qty: {draft.qty:.6g}\n"
            f"- Price: {self.bot._fmt_price(draft.price)} (maker)"
        )
        if interaction.response.is_done():
            message = await interaction.followup.send(text, view=view, ephemeral=True)
        else:
            await interaction.response.send_message(text, view=view, ephemeral=True)
            message = await interaction.original_response()
        view.message = message


class ClosePositionSelectView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        drafts: List[ClosePositionDraft],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.message: Optional[discord.Message] = None
        self.add_item(ClosePositionSelect(bot, drafts, requester_id))

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)


class ClosePositionConfirmView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        requester_id: int,
        draft: ClosePositionDraft,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.draft = draft
        self.message: Optional[discord.Message] = None

    @discord.ui.button(label="Close Position", style=discord.ButtonStyle.danger)
    async def close(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can close this position.",
                ephemeral=True,
            )
            return
        if not interaction.response.is_done():
            try:
                await interaction.response.defer(ephemeral=True, thinking=True)
            except (discord.NotFound, discord.errors.NotFound, discord.HTTPException):
                pass
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)
        await self.bot._execute_close_position(interaction, self.draft)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the requester can cancel this.",
                ephemeral=True,
            )
            return
        for item in self.children:
            item.disabled = True
        if interaction.response.is_done():
            await interaction.followup.send("Cancelled.", ephemeral=True)
        else:
            await interaction.response.send_message("Cancelled.", ephemeral=True)
        if self.message:
            await self.message.edit(view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)

class SettingSaveChoiceView(View):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.message: Optional[discord.Message] = None

    async def _reject_if_not_owner(
        self, interaction: discord.Interaction
    ) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the save can use these buttons.",
                ephemeral=True,
            )
            return False
        return True

    @discord.ui.button(label="Save New", style=discord.ButtonStyle.primary)
    async def save_new(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if not await self._reject_if_not_owner(interaction):
            return
        await interaction.response.send_modal(
            SettingSaveModal(self.bot, self.requester_id)
        )

    @discord.ui.button(label="Overwrite Existing", style=discord.ButtonStyle.secondary)
    async def overwrite(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if not await self._reject_if_not_owner(interaction):
            return
        profiles = self.bot._list_setting_profiles()
        if not profiles:
            await interaction.response.send_message(
                "No saved profiles to overwrite.", ephemeral=True
            )
            return
        view = SettingOverwriteView(self.bot, profiles, self.requester_id)
        if interaction.response.is_done():
            await interaction.followup.send(
                "Select a profile to overwrite with the current settings.",
                view=view,
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "Select a profile to overwrite with the current settings.",
                view=view,
                ephemeral=True,
            )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if not await self._reject_if_not_owner(interaction):
            return
        for item in self.children:
            item.disabled = True
        if interaction.response.is_done():
            await interaction.followup.send("Cancelled.", ephemeral=True)
        else:
            await interaction.response.send_message("Cancelled.", ephemeral=True)
        if self.message:
            await self.message.edit(view=self)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)


class DeleteDataSelect(Select):
    def __init__(self, bot: "BacktestBot", files: List[str]) -> None:
        self.bot = bot
        options = [
            discord.SelectOption(label=name[:100], value=name) for name in files
        ]
        super().__init__(
            placeholder="Select a data file to delete",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        filename = self.values[0]
        deleted, message = await self.bot._delete_data_file(filename)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
        if deleted:
            await self.bot._update_data_list_channel()
            await self.bot._update_strategy_list_channel()


class DeleteDataView(View):
    def __init__(self, bot: "BacktestBot", files: List[str]) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.add_item(DeleteDataSelect(bot, files))


class BacktestDataSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.data_overrides = data_overrides
        self.backtest_overrides = backtest_overrides
        self.strategy_overrides = strategy_overrides
        self.requester_id = requester_id
        options = [
            discord.SelectOption(label=name[:100], value=name) for name in files
        ]
        super().__init__(
            placeholder="Select a data file to run backtest",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the backtest can select a file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.", ephemeral=True
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        data_overrides = dict(self.data_overrides)
        data_overrides["file_pattern"] = self.values[0]
        strategy_overrides, err = self.bot._enforce_cheatkey_strategy(
            interaction.channel,
            self.strategy_overrides,
        )
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            message, chart_paths = await self.bot._execute_backtest(
                data_overrides,
                self.backtest_overrides,
                strategy_overrides,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Backtest failed: {exc}", ephemeral=True
            )
            return
        await self.bot._send_backtest_result(
            message,
            interaction,
            chart_paths,
        )
        await self.bot._send_rerun_prompt(
            interaction,
            RerunBacktestView(
                self.bot,
                self.requester_id,
                data_overrides,
                self.backtest_overrides,
                strategy_overrides,
            ),
        )


class EntryDiffDataSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
        diff_config: EntryDiffConfig,
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.data_overrides = data_overrides
        self.backtest_overrides = backtest_overrides
        self.strategy_overrides = strategy_overrides
        self.diff_config = diff_config
        self.requester_id = requester_id
        options = [discord.SelectOption(label=name[:100], value=name) for name in files]
        super().__init__(
            placeholder="Select a data file to run entry diff",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the analysis can select a file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run this analysis.", ephemeral=True
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        data_overrides = dict(self.data_overrides)
        data_overrides["file_pattern"] = self.values[0]
        strategy_overrides, err = self.bot._enforce_cheatkey_strategy(
            interaction.channel,
            self.strategy_overrides,
        )
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            message, chart_paths = await self.bot._execute_entry_diff(
                data_overrides,
                self.backtest_overrides,
                strategy_overrides,
                self.diff_config,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Entry diff failed: {exc}", ephemeral=True
            )
            return
        await self.bot._send_backtest_result(message, interaction, chart_paths)


class EntryDiffDataView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
        diff_config: EntryDiffConfig,
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.add_item(
            EntryDiffDataSelect(
                bot,
                files,
                data_overrides,
                backtest_overrides,
                strategy_overrides,
                diff_config,
                requester_id,
            )
        )


class BacktestDataView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.add_item(
            BacktestDataSelect(
                bot,
                files,
                data_overrides,
                backtest_overrides,
                strategy_overrides,
                requester_id,
            )
        )


class RerunBacktestView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        requester_id: int,
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.requester_id = requester_id
        self.data_overrides = data_overrides
        self.backtest_overrides = backtest_overrides
        self.strategy_overrides = strategy_overrides

    @discord.ui.button(label="Run again", style=discord.ButtonStyle.primary)
    async def rerun_backtest(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button,
    ) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who started the backtest can rerun it.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This button can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return

        result_channel = await self.bot._get_result_channel()
        is_allowed_channel = self.bot._is_backtest_channel(interaction.channel)
        if result_channel and interaction.channel:
            is_allowed_channel = is_allowed_channel or (
                interaction.channel.id == result_channel.id
            )
        if not is_allowed_channel:
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        strategy_overrides, err = self.bot._enforce_cheatkey_strategy(
            interaction.channel,
            self.strategy_overrides,
        )
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            message, chart_paths = await self.bot._execute_backtest(
                self.data_overrides,
                self.backtest_overrides,
                strategy_overrides,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Backtest failed: {exc}",
                ephemeral=True,
            )
            return
        await self.bot._send_backtest_result(
            message,
            interaction,
            chart_paths,
        )
        await self.bot._send_rerun_prompt(
            interaction,
            RerunBacktestView(
                self.bot,
                self.requester_id,
                self.data_overrides,
                self.backtest_overrides,
                strategy_overrides,
            ),
        )


class GridRangeModal(Modal):
    def __init__(self, view: "GridSetupView", index: int) -> None:
        title = "Set range for variable 1" if index == 1 else "Set range for variable 2"
        super().__init__(title=title)
        self.view = view
        self.index = index
        existing = view.range1 if index == 1 else view.range2
        start_default = existing[0] if existing else ""
        end_default = existing[1] if existing else ""
        step_default = existing[2] if existing else ""

        self.start_input = TextInput(
            label="Start",
            default=start_default,
            required=True,
            max_length=32,
        )
        self.end_input = TextInput(
            label="End",
            default=end_default,
            required=True,
            max_length=32,
        )
        self.step_input = TextInput(
            label="Step",
            default=step_default,
            required=True,
            max_length=32,
        )
        self.add_item(self.start_input)
        self.add_item(self.end_input)
        self.add_item(self.step_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        values = (
            self.start_input.value.strip(),
            self.end_input.value.strip(),
            self.step_input.value.strip(),
        )
        if self.index == 1:
            self.view.range1 = values
        else:
            self.view.range2 = values
        await self.view.refresh_message(interaction)


class GridVariableSelect(Select):
    def __init__(self, view: "GridSetupView", index: int) -> None:
        self.grid_view = view
        self.index = index
        placeholder = "Select variable 1" if index == 1 else "Select variable 2 (or None)"
        super().__init__(
            placeholder=placeholder,
            min_values=1,
            max_values=1,
            options=view.options_for_page(index),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.grid_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the grid can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.grid_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.grid_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.grid_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        if self.index == 1:
            self.grid_view.var1_key = selected
        else:
            self.grid_view.var2_key = None if selected == "__none__" else selected
            if self.grid_view.var2_key is None:
                self.grid_view.range2 = None
        await self.grid_view.refresh_message(interaction)


class GridDataFileSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.tokens = tokens
        self.requester_id = requester_id
        options = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=True,
            )
        ]
        for filename in files:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                )
            )
        super().__init__(
            placeholder="Select data file for grid",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the grid can select a data file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        data_file = None if selected == "__default__" else selected
        await self.bot._run_grid_from_interaction(
            interaction,
            self.tokens,
            data_file,
        )


class GridDataFileView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(GridDataFileSelect(bot, files, tokens, requester_id))


class CustomDataFileSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.tokens = tokens
        self.requester_id = requester_id
        options = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=True,
            )
        ]
        for filename in files:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                )
            )
        super().__init__(
            placeholder="Select data file for costom",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the custom run can select a data file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        data_file = None if selected == "__default__" else selected
        await self.bot._run_costom_from_interaction(
            interaction,
            self.tokens,
            data_file,
        )


class CustomDataFileView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(CustomDataFileSelect(bot, files, tokens, requester_id))


class Costom2DataFileSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.tokens = tokens
        self.requester_id = requester_id
        options = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=True,
            )
        ]
        for filename in files:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                )
            )
        super().__init__(
            placeholder="Select data file for costom2",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom2 run can select a data file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        data_file = None if selected == "__default__" else selected
        await self.bot._run_costom2_from_interaction(
            interaction,
            self.tokens,
            data_file,
        )


class Costom2DataFileView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(Costom2DataFileSelect(bot, files, tokens, requester_id))


class Costom4DataFileSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.tokens = tokens
        self.requester_id = requester_id
        options = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=True,
            )
        ]
        for filename in files:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                )
            )
        super().__init__(
            placeholder="Select data file for costom4",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom4 run can select a data file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        data_file = None if selected == "__default__" else selected
        await self.bot._run_costom4_from_interaction(
            interaction,
            self.tokens,
            data_file,
        )


class Costom4DataFileView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(Costom4DataFileSelect(bot, files, tokens, requester_id))


class Costom3DataFileSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.tokens = tokens
        self.requester_id = requester_id
        options = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=True,
            )
        ]
        for filename in files:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                )
            )
        super().__init__(
            placeholder="Select data file for costom3",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom3 run can select a data file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        data_file = None if selected == "__default__" else selected
        await self.bot._run_costom3_from_interaction(
            interaction,
            self.tokens,
            data_file,
        )


class Costom3DataFileView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(Costom3DataFileSelect(bot, files, tokens, requester_id))


class GridDataSelect(Select):
    def __init__(self, view: "GridSetupView") -> None:
        self.grid_view = view
        super().__init__(
            placeholder="Select data file (optional)",
            min_values=1,
            max_values=1,
            options=view.data_options(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.grid_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the grid can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.grid_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.grid_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.grid_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        self.grid_view.data_file = None if selected == "__default__" else selected
        await self.grid_view.refresh_message(interaction)


class CompRangeModal(Modal):
    def __init__(self, view: "CompSetupView") -> None:
        super().__init__(title="Set range for variable")
        self.view = view
        existing = view.range
        start_default = existing[0] if existing else ""
        end_default = existing[1] if existing else ""
        step_default = existing[2] if existing else ""

        self.start_input = TextInput(
            label="Start",
            default=start_default,
            required=True,
            max_length=32,
        )
        self.end_input = TextInput(
            label="End",
            default=end_default,
            required=True,
            max_length=32,
        )
        self.step_input = TextInput(
            label="Step",
            default=step_default,
            required=True,
            max_length=32,
        )
        self.add_item(self.start_input)
        self.add_item(self.end_input)
        self.add_item(self.step_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        self.view.range = (
            self.start_input.value.strip(),
            self.end_input.value.strip(),
            self.step_input.value.strip(),
        )
        await self.view.refresh_message(interaction)


class CompVariableSelect(Select):
    def __init__(self, view: "CompSetupView") -> None:
        self.comp_view = view
        super().__init__(
            placeholder="Select variable",
            min_values=1,
            max_values=1,
            options=view.options_for_page(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.comp_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the comp can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.comp_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.comp_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.comp_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        self.comp_view.var_key = self.values[0]
        await self.comp_view.refresh_message(interaction)


class CompDataSelect(Select):
    def __init__(self, view: "CompSetupView") -> None:
        self.comp_view = view
        super().__init__(
            placeholder="Select data file (optional)",
            min_values=1,
            max_values=1,
            options=view.data_options(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.comp_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the comp can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.comp_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.comp_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.comp_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        self.comp_view.data_file = None if selected == "__default__" else selected
        await self.comp_view.refresh_message(interaction)


class CompDataFileSelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.tokens = tokens
        self.requester_id = requester_id
        options = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=True,
            )
        ]
        for filename in files:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                )
            )
        super().__init__(
            placeholder="Select data file for comp",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the comp can select a data file.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        data_file = None if selected == "__default__" else selected
        await self.bot._run_comp_from_interaction(
            interaction,
            self.tokens,
            data_file,
        )


class CompDataFileView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        files: List[str],
        tokens: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(CompDataFileSelect(bot, files, tokens, requester_id))


class CompSetupView(View):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(timeout=600)
        self.bot = bot
        self.requester_id = requester_id
        self.keys = bot._grid_variable_keys()
        self.page = 0
        self.page_size = 24
        self.data_file: Optional[str] = None
        self.data_files = bot._list_price_files()
        self.max_data_options = 24
        self.var_key: Optional[str] = None
        self.range: Optional[tuple[str, str, str]] = None
        self.message: Optional[discord.Message] = None

        self.select_data = CompDataSelect(self)
        self.select_var = CompVariableSelect(self)
        self.prev_button = Button(label="Prev vars", style=discord.ButtonStyle.secondary)
        self.next_button = Button(label="Next vars", style=discord.ButtonStyle.secondary)
        self.range_button = Button(label="Set range", style=discord.ButtonStyle.secondary)
        self.run_button = Button(label="Run comp", style=discord.ButtonStyle.primary)

        self.prev_button.callback = self.prev_page
        self.next_button.callback = self.next_page
        self.range_button.callback = self.set_range
        self.run_button.callback = self.run_comp

        self.add_item(self.select_data)
        self.add_item(self.select_var)
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self.add_item(self.range_button)
        self.add_item(self.run_button)
        self._update_buttons()

    def _update_buttons(self) -> None:
        max_pages = max(1, (len(self.keys) - 1) // self.page_size + 1)
        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= max_pages - 1
        self.run_button.disabled = self.var_key is None or self.range is None

    def _page_keys(self) -> List[str]:
        start = self.page * self.page_size
        end = start + self.page_size
        return self.keys[start:end]

    def data_options(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=self.data_file is None,
            )
        ]
        for filename in self.data_files[: self.max_data_options]:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                    default=self.data_file == filename,
                )
            )
        if len(options) == 1:
            options[0] = discord.SelectOption(
                label="Default (no data files found)",
                value="__default__",
                default=True,
            )
        return options

    def options_for_page(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = []
        for key in self._page_keys():
            selected = self.var_key == key
            options.append(
                discord.SelectOption(
                    label=key[:100],
                    value=key,
                    default=selected,
                )
            )
        if not options:
            options.append(
                discord.SelectOption(label="No numeric variables", value="__none__")
            )
        return options

    def _summary(self) -> str:
        def fmt_range(value: Optional[tuple[str, str, str]]) -> str:
            if not value:
                return "not set"
            return f"{value[0]} → {value[1]} (step {value[2]})"

        data_label = self.data_file or "default (settings data.file_pattern)"
        var_text = self.var_key or "not selected"
        lines = [
            "Select variable and range for comp backtest.",
            f"- data file: {data_label}",
            f"- variable: {var_text}",
            f"- range: {fmt_range(self.range)}",
        ]
        if len(self.data_files) > self.max_data_options:
            lines.append(
                f"- data list truncated: showing {self.max_data_options}/{len(self.data_files)} "
                "(use !data list or !comp --data FILENAME)"
            )
        return "\n".join(lines)

    async def refresh_message(self, interaction: discord.Interaction) -> None:
        self.select_data.options = self.data_options()
        self.select_var.options = self.options_for_page()
        self._update_buttons()
        content = self._summary()
        if interaction.message:
            await interaction.response.edit_message(content=content, view=self)
            return
        await interaction.response.defer()
        if self.message:
            await self.message.edit(content=content, view=self)

    async def prev_page(self, interaction: discord.Interaction) -> None:
        if self.page > 0:
            self.page -= 1
        await self.refresh_message(interaction)

    async def next_page(self, interaction: discord.Interaction) -> None:
        max_pages = max(1, (len(self.keys) - 1) // self.page_size + 1)
        if self.page < max_pages - 1:
            self.page += 1
        await self.refresh_message(interaction)

    async def set_range(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the comp can change ranges.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(CompRangeModal(self))

    async def run_comp(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the comp can run it.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return
        if self.var_key is None or self.range is None:
            await interaction.response.send_message(
                "Select a variable and set its range first.",
                ephemeral=True,
            )
            return

        await self.bot._run_comp_from_interaction(
            interaction,
            [self.var_key, *self.range],
            self.data_file,
        )

    async def on_timeout(self) -> None:
        self.select_data.disabled = True
        self.select_var.disabled = True
        self.prev_button.disabled = True
        self.next_button.disabled = True
        self.range_button.disabled = True
        self.run_button.disabled = True
        if self.message:
            await self.message.edit(view=self)


class Costom3VariableSelect(Select):
    def __init__(self, view: "Costom3SetupView") -> None:
        self.costom3_view = view
        super().__init__(
            placeholder="Select variable",
            min_values=1,
            max_values=1,
            options=view.options_for_page(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.costom3_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom3 can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.costom3_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.costom3_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.costom3_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        self.costom3_view.var_key = self.values[0]
        await self.costom3_view.refresh_message(interaction)


class Costom3DataSelect(Select):
    def __init__(self, view: "Costom3SetupView") -> None:
        self.costom3_view = view
        super().__init__(
            placeholder="Select data file (optional)",
            min_values=1,
            max_values=1,
            options=view.data_options(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.costom3_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom3 can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.costom3_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.costom3_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.costom3_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        self.costom3_view.data_file = None if selected == "__default__" else selected
        await self.costom3_view.refresh_message(interaction)


class Costom3SetupView(View):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(timeout=600)
        self.bot = bot
        self.requester_id = requester_id
        self.keys = bot._grid_variable_keys()
        self.page = 0
        self.page_size = 24
        self.data_file: Optional[str] = None
        self.data_files = bot._list_price_files()
        self.max_data_options = 24
        self.var_key: Optional[str] = None
        self.range: Optional[tuple[str, str, str]] = None
        self.message: Optional[discord.Message] = None

        self.select_data = Costom3DataSelect(self)
        self.select_var = Costom3VariableSelect(self)
        self.prev_button = Button(label="Prev vars", style=discord.ButtonStyle.secondary)
        self.next_button = Button(label="Next vars", style=discord.ButtonStyle.secondary)
        self.range_button = Button(label="Set range", style=discord.ButtonStyle.secondary)
        self.run_button = Button(label="Run costom3", style=discord.ButtonStyle.primary)

        self.prev_button.callback = self.prev_page
        self.next_button.callback = self.next_page
        self.range_button.callback = self.set_range
        self.run_button.callback = self.run_costom3

        self.add_item(self.select_data)
        self.add_item(self.select_var)
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self.add_item(self.range_button)
        self.add_item(self.run_button)
        self._update_buttons()

    def _update_buttons(self) -> None:
        max_pages = max(1, (len(self.keys) - 1) // self.page_size + 1)
        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= max_pages - 1
        self.run_button.disabled = self.var_key is None or self.range is None

    def _page_keys(self) -> List[str]:
        start = self.page * self.page_size
        end = start + self.page_size
        return self.keys[start:end]

    def data_options(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=self.data_file is None,
            )
        ]
        for filename in self.data_files[: self.max_data_options]:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                    default=self.data_file == filename,
                )
            )
        if len(options) == 1:
            options[0] = discord.SelectOption(
                label="Default (no data files found)",
                value="__default__",
                default=True,
            )
        return options

    def options_for_page(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = []
        for key in self._page_keys():
            selected = self.var_key == key
            options.append(
                discord.SelectOption(
                    label=key[:100],
                    value=key,
                    default=selected,
                )
            )
        if not options:
            options.append(
                discord.SelectOption(label="No numeric variables", value="__none__")
            )
        return options

    def _summary(self) -> str:
        def fmt_range(value: Optional[tuple[str, str, str]]) -> str:
            if not value:
                return "not set"
            return f"{value[0]} → {value[1]} (step {value[2]})"

        data_label = self.data_file or "default (settings data.file_pattern)"
        var_text = self.var_key or "not selected"
        lines = [
            "Select variable and range for costom3 backtest.",
            f"- data file: {data_label}",
            f"- variable: {var_text}",
            f"- range: {fmt_range(self.range)}",
        ]
        if len(self.data_files) > self.max_data_options:
            lines.append(
                f"- data list truncated: showing {self.max_data_options}/{len(self.data_files)} "
                "(use !data list or !costom3 --data FILENAME)"
            )
        return "\n".join(lines)

    async def refresh_message(self, interaction: discord.Interaction) -> None:
        self.select_data.options = self.data_options()
        self.select_var.options = self.options_for_page()
        self._update_buttons()
        content = self._summary()
        if interaction.message:
            await interaction.response.edit_message(content=content, view=self)
            return
        await interaction.response.defer()
        if self.message:
            await self.message.edit(content=content, view=self)

    async def prev_page(self, interaction: discord.Interaction) -> None:
        if self.page > 0:
            self.page -= 1
        await self.refresh_message(interaction)

    async def next_page(self, interaction: discord.Interaction) -> None:
        max_pages = max(1, (len(self.keys) - 1) // self.page_size + 1)
        if self.page < max_pages - 1:
            self.page += 1
        await self.refresh_message(interaction)

    async def set_range(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom3 can change ranges.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(CompRangeModal(self))

    async def run_costom3(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the costom3 can run it.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return
        if self.var_key is None or self.range is None:
            await interaction.response.send_message(
                "Select a variable and set its range first.",
                ephemeral=True,
            )
            return

        await self.bot._run_costom3_from_interaction(
            interaction,
            [self.var_key, *self.range],
            self.data_file,
        )

    async def on_timeout(self) -> None:
        self.select_data.disabled = True
        self.select_var.disabled = True
        self.prev_button.disabled = True
        self.next_button.disabled = True
        self.range_button.disabled = True
        self.run_button.disabled = True
        if self.message:
            await self.message.edit(view=self)

class CustomParamsModal(Modal):
    def __init__(self, view: "CustomSetupView") -> None:
        super().__init__(title="Set pre-low/high check params")
        self.view = view
        lookback_default = view.lookback_value or ""
        buffer_default = view.buffer_value or ""
        self.lookback_input = TextInput(
            label="Lookback bars (N)",
            default=lookback_default,
            required=True,
            max_length=16,
        )
        self.buffer_input = TextInput(
            label="Buffer percent (e.g. 0.1 for 0.1%)",
            default=buffer_default,
            required=True,
            max_length=16,
        )
        self.add_item(self.lookback_input)
        self.add_item(self.buffer_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        self.view.lookback_value = self.lookback_input.value.strip()
        self.view.buffer_value = self.buffer_input.value.strip()
        await self.view.refresh_message(interaction)


class CustomDataSelect(Select):
    def __init__(self, view: "CustomSetupView") -> None:
        self.custom_view = view
        super().__init__(
            placeholder="Select data file (optional)",
            min_values=1,
            max_values=1,
            options=view.data_options(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.custom_view.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the custom run can change selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.custom_view.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.custom_view.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.custom_view.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        selected = self.values[0]
        self.custom_view.data_file = None if selected == "__default__" else selected
        await self.custom_view.refresh_message(interaction)


class CustomSetupView(View):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(timeout=600)
        self.bot = bot
        self.requester_id = requester_id
        self.data_file: Optional[str] = None
        self.data_files = bot._list_price_files()
        self.max_data_options = 24
        self.lookback_value: Optional[str] = None
        self.buffer_value: Optional[str] = None
        self.message: Optional[discord.Message] = None

        self.select_data = CustomDataSelect(self)
        self.params_button = Button(label="Set params", style=discord.ButtonStyle.secondary)
        self.run_button = Button(label="Run costom", style=discord.ButtonStyle.primary)

        self.params_button.callback = self.set_params
        self.run_button.callback = self.run_costom

        self.add_item(self.select_data)
        self.add_item(self.params_button)
        self.add_item(self.run_button)
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.run_button.disabled = self.lookback_value is None or self.buffer_value is None

    def data_options(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=self.data_file is None,
            )
        ]
        for filename in self.data_files[: self.max_data_options]:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                    default=self.data_file == filename,
                )
            )
        if len(options) == 1:
            options[0] = discord.SelectOption(
                label="Default (no data files found)",
                value="__default__",
                default=True,
            )
        return options

    def _summary(self) -> str:
        data_label = self.data_file or "default (settings data.file_pattern)"
        lookback_text = self.lookback_value or "not set"
        buffer_text = self.buffer_value or "not set"
        lines = [
            "Set lookback + buffer for pre-low/high check.",
            f"- data file: {data_label}",
            f"- lookback bars: {lookback_text}",
            f"- buffer (%): {buffer_text}",
        ]
        if len(self.data_files) > self.max_data_options:
            lines.append(
                f"- data list truncated: showing {self.max_data_options}/{len(self.data_files)} "
                "(use !data list or !costom --data FILENAME)"
            )
        return "\n".join(lines)

    async def refresh_message(self, interaction: discord.Interaction) -> None:
        self.select_data.options = self.data_options()
        self._update_buttons()
        content = self._summary()
        if interaction.message:
            await interaction.response.edit_message(content=content, view=self)
            return
        await interaction.response.defer()
        if self.message:
            await self.message.edit(content=content, view=self)

    async def set_params(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the custom run can change params.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(CustomParamsModal(self))

    async def run_costom(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the custom run can run it.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return
        if self.lookback_value is None or self.buffer_value is None:
            await interaction.response.send_message(
                "Set lookback and buffer first.",
                ephemeral=True,
            )
            return

        await self.bot._run_costom_from_interaction(
            interaction,
            [self.lookback_value, self.buffer_value],
            self.data_file,
        )

    async def on_timeout(self) -> None:
        self.select_data.disabled = True
        self.params_button.disabled = True
        self.run_button.disabled = True
        if self.message:
            await self.message.edit(view=self)


class GridSetupView(View):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(timeout=600)
        self.bot = bot
        self.requester_id = requester_id
        self.keys = bot._grid_variable_keys()
        self.page = 0
        self.page_size = 24
        self.data_file: Optional[str] = None
        self.data_files = bot._list_price_files()
        self.max_data_options = 24
        self.var1_key: Optional[str] = None
        self.var2_key: Optional[str] = None
        self.range1: Optional[tuple[str, str, str]] = None
        self.range2: Optional[tuple[str, str, str]] = None
        self.message: Optional[discord.Message] = None

        self.select_data = GridDataSelect(self)
        self.select_var1 = GridVariableSelect(self, 1)
        self.select_var2 = GridVariableSelect(self, 2)
        self.prev_button = Button(label="Prev vars", style=discord.ButtonStyle.secondary)
        self.next_button = Button(label="Next vars", style=discord.ButtonStyle.secondary)
        self.range1_button = Button(label="Set range 1", style=discord.ButtonStyle.secondary)
        self.range2_button = Button(label="Set range 2", style=discord.ButtonStyle.secondary)
        self.run_button = Button(label="Run grid", style=discord.ButtonStyle.primary)

        self.prev_button.callback = self.prev_page
        self.next_button.callback = self.next_page
        self.range1_button.callback = self.set_range1
        self.range2_button.callback = self.set_range2
        self.run_button.callback = self.run_grid

        self.add_item(self.select_data)
        self.add_item(self.select_var1)
        self.add_item(self.select_var2)
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self.add_item(self.range1_button)
        self.add_item(self.range2_button)
        self.add_item(self.run_button)
        self._update_buttons()

    def _update_buttons(self) -> None:
        max_pages = max(1, (len(self.keys) - 1) // self.page_size + 1)
        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= max_pages - 1
        self.run_button.disabled = self.var1_key is None or self.range1 is None

    def _page_keys(self) -> List[str]:
        start = self.page * self.page_size
        end = start + self.page_size
        return self.keys[start:end]

    def data_options(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = [
            discord.SelectOption(
                label="Default (use settings data.file_pattern)",
                value="__default__",
                default=self.data_file is None,
            )
        ]
        for filename in self.data_files[: self.max_data_options]:
            options.append(
                discord.SelectOption(
                    label=filename[:100],
                    value=filename,
                    default=self.data_file == filename,
                )
            )
        if len(options) == 1:
            options[0] = discord.SelectOption(
                label="Default (no data files found)",
                value="__default__",
                default=True,
            )
        return options

    def options_for_page(self, index: int) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = []
        if index == 2:
            options.append(
                discord.SelectOption(
                    label="None (single variable)",
                    value="__none__",
                    default=self.var2_key is None,
                )
            )
        for key in self._page_keys():
            selected = (self.var1_key == key) if index == 1 else (self.var2_key == key)
            options.append(
                discord.SelectOption(
                    label=key[:100],
                    value=key,
                    default=selected,
                )
            )
        if not options:
            options.append(
                discord.SelectOption(label="No numeric variables", value="__none__")
            )
        return options

    def _summary(self) -> str:
        def fmt_range(value: Optional[tuple[str, str, str]]) -> str:
            if not value:
                return "not set"
            return f"{value[0]} → {value[1]} (step {value[2]})"

        data_label = self.data_file or "default (settings data.file_pattern)"
        var1_text = self.var1_key or "not selected"
        var2_text = self.var2_key if self.var2_key is not None else "None"
        lines = [
            "Select variables and ranges for grid backtest.",
            f"- data file: {data_label}",
            f"- variable 1: {var1_text}",
            f"- range 1: {fmt_range(self.range1)}",
            f"- variable 2: {var2_text}",
            f"- range 2: {fmt_range(self.range2)}",
        ]
        if len(self.data_files) > self.max_data_options:
            lines.append(
                f"- data list truncated: showing {self.max_data_options}/{len(self.data_files)} "
                "(use !data list or !grid --data FILENAME)"
            )
        return "\n".join(lines)

    async def refresh_message(self, interaction: discord.Interaction) -> None:
        self.select_data.options = self.data_options()
        self.select_var1.options = self.options_for_page(1)
        self.select_var2.options = self.options_for_page(2)
        self._update_buttons()
        content = self._summary()
        if interaction.message:
            await interaction.response.edit_message(content=content, view=self)
            return
        await interaction.response.defer()
        if self.message:
            await self.message.edit(content=content, view=self)

    async def prev_page(self, interaction: discord.Interaction) -> None:
        if self.page > 0:
            self.page -= 1
        await self.refresh_message(interaction)

    async def next_page(self, interaction: discord.Interaction) -> None:
        max_pages = max(1, (len(self.keys) - 1) // self.page_size + 1)
        if self.page < max_pages - 1:
            self.page += 1
        await self.refresh_message(interaction)

    async def set_range1(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the grid can change ranges.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(GridRangeModal(self, 1))

    async def set_range2(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the grid can change ranges.",
                ephemeral=True,
            )
            return
        if self.var2_key is None:
            await interaction.response.send_message(
                "Select a second variable before setting range 2.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(GridRangeModal(self, 2))

    async def run_grid(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the grid can run it.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.",
                ephemeral=True,
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.",
                ephemeral=True,
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return
        if self.var1_key is None or self.range1 is None:
            await interaction.response.send_message(
                "Select variable 1 and set its range first.",
                ephemeral=True,
            )
            return
        if self.var2_key is not None and self.range2 is None:
            await interaction.response.send_message(
                "Set range 2 or choose None for variable 2.",
                ephemeral=True,
            )
            return

        base_settings = load_settings(self.bot.settings_path)
        if self.data_file:
            base_settings.data["file_pattern"] = self.data_file
        err = self.bot._apply_forced_strategy(base_settings, interaction.channel)
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        var1, err = self.bot._parse_grid_var(base_settings, self.var1_key)
        if err:
            await interaction.response.send_message(
                f"Invalid first variable: {err}",
                ephemeral=True,
            )
            return
        values1, err = self.bot._build_grid_values(*self.range1)
        if err:
            await interaction.response.send_message(
                f"Invalid first range: {err}",
                ephemeral=True,
            )
            return

        var2: Optional[dict] = None
        values2: Optional[List[Any]] = None
        if self.var2_key is not None:
            var2, err = self.bot._parse_grid_var(base_settings, self.var2_key)
            if err:
                await interaction.response.send_message(
                    f"Invalid second variable: {err}",
                    ephemeral=True,
                )
                return
            if self.range2 is None:
                await interaction.response.send_message(
                    "Set range 2 or choose None for variable 2.",
                    ephemeral=True,
                )
                return
            values2, err = self.bot._build_grid_values(*self.range2)
            if err:
                await interaction.response.send_message(
                    f"Invalid second range: {err}",
                    ephemeral=True,
                )
                return

        max_runs = 250
        total_runs = len(values1) * (len(values2) if values2 else 1)
        if total_runs > max_runs:
            await interaction.response.send_message(
                f"Too many combinations ({total_runs}). Please keep it under {max_runs}.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        try:
            progress_msg = await interaction.followup.send(
                f"Running grid backtests (0/{total_runs})...",
                wait=True,
            )
        except discord.HTTPException as exc:
            logger.warning("Grid followup send failed; falling back to channel: %s", exc)
            if interaction.channel is None:
                raise
            progress_msg = await interaction.channel.send(
                f"Running grid backtests (0/{total_runs})..."
            )
        progress_channel = progress_msg.channel or interaction.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(interaction.channel.id)
        )
        progress_handle = self.bot._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running grid backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Grid progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.bot.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Grid progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running grid backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Grid progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self.bot._run_grid_backtests,
                base_settings,
                var1,
                values1,
                var2,
                values2,
                progress_cb,
                total_runs,
            )
        finally:
            self.bot._unregister_grid_progress(progress_handle)
        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        output = self.bot._format_grid_output(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )
        image_paths = self.bot._build_grid_heatmap_images(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )
        data_paths = self.bot._build_grid_data_file(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )

        result_channel = await self.bot._get_result_channel()
        target_channel = result_channel or interaction.channel or progress_channel
        if target_channel is None:
            await interaction.followup.send(
                "Grid backtest complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return
        sent = await self.bot._send_grid_output_to_channel(
            target_channel.id,
            output,
            image_paths=image_paths,
            data_paths=data_paths,
        )
        if not sent:
            self.bot._queue_grid_output(
                target_channel.id,
                output,
                image_paths=image_paths,
                data_paths=data_paths,
            )
        if result_channel is not None:
            try:
                await interaction.followup.send(
                    f"Grid backtest complete. Results posted in #{result_channel.name}.",
                    ephemeral=True,
                )
            except discord.HTTPException as exc:
                logger.warning("Grid followup send failed: %s", exc)
                if target_channel is not None:
                    await target_channel.send(
                        f"Grid backtest complete. Results posted in #{result_channel.name}."
                    )
        else:
            try:
                await interaction.followup.send(
                    "Grid backtest complete. Results posted here.",
                    ephemeral=True,
                )
            except discord.HTTPException as exc:
                logger.warning("Grid followup send failed: %s", exc)
                if target_channel is not None:
                    await target_channel.send("Grid backtest complete. Results posted here.")

    async def on_timeout(self) -> None:
        self.select_data.disabled = True
        self.select_var1.disabled = True
        self.select_var2.disabled = True
        self.prev_button.disabled = True
        self.next_button.disabled = True
        self.range1_button.disabled = True
        self.range2_button.disabled = True
        self.run_button.disabled = True
        if self.message:
            await self.message.edit(view=self)


class StrategySelect(Select):
    def __init__(
        self,
        bot: "BacktestBot",
        strategies: List[str],
        requester_id: int,
    ) -> None:
        self.bot = bot
        self.requester_id = requester_id
        options = [
            discord.SelectOption(label=name[:100], value=name) for name in strategies
        ]
        super().__init__(
            placeholder="Select a strategy",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the change can select a strategy.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can change strategies.", ephemeral=True
            )
            return
        if not self.bot._is_command_channel(interaction.channel):
            await interaction.response.send_message(
                f"Please use #{self.bot.command_channel_name} for strategy changes.",
                ephemeral=True,
            )
            return

        name = self.values[0]
        self.bot._apply_setting("strategy", "name", name)
        save_settings(self.bot.settings_path, self.bot.settings)
        await interaction.response.send_message(
            f"Strategy set to {name}.", ephemeral=True
        )


class StrategySelectView(View):
    def __init__(
        self,
        bot: "BacktestBot",
        strategies: List[str],
        requester_id: int,
    ) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.add_item(StrategySelect(bot, strategies, requester_id))


class ChartTypeSelect(Select):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        self.bot = bot
        self.requester_id = requester_id
        options = [
            discord.SelectOption(label="Profit (top winners)", value="profit"),
            discord.SelectOption(label="Loss (top losers)", value="loss"),
        ]
        super().__init__(
            placeholder="Select profit or loss trades",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the charts can make selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can request charts.", ephemeral=True
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        choice = self.values[0]
        view = ChartCountView(self.bot, choice, self.requester_id)
        await interaction.response.edit_message(
            content="Select how many trades to show (1-10).",
            view=view,
        )


class ChartTypeView(View):
    def __init__(self, bot: "BacktestBot", requester_id: int) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.add_item(ChartTypeSelect(bot, requester_id))


class ChartCountSelect(Select):
    def __init__(self, bot: "BacktestBot", mode: str, requester_id: int) -> None:
        self.bot = bot
        self.mode = mode
        self.requester_id = requester_id
        options = [
            discord.SelectOption(label=str(count), value=str(count))
            for count in range(1, 11)
        ]
        super().__init__(
            placeholder="Select trade count",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the user who requested the charts can make selections.",
                ephemeral=True,
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self.bot._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can request charts.", ephemeral=True
            )
            return
        if not self.bot._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self.bot._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        if self.bot._last_backtest is None:
            await interaction.response.send_message(
                "Run `!testrun` first so I have results to chart.",
                ephemeral=True,
            )
            return

        count = int(self.values[0])
        await interaction.response.defer(thinking=True)
        await self.bot._send_trade_charts(interaction, self.mode, count)


class ChartCountView(View):
    def __init__(self, bot: "BacktestBot", mode: str, requester_id: int) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.add_item(ChartCountSelect(bot, mode, requester_id))


class ConfigSelect(Select):
    def __init__(self, editor: "ConfigEditorView") -> None:
        self.editor = editor
        super().__init__(
            placeholder="Select a variable to edit",
            min_values=1,
            max_values=1,
            options=editor.options_for_page(),
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        full_key = self.values[0]
        if full_key == "__none__":
            await interaction.response.send_message(
                "No editable values available.", ephemeral=True
            )
            return
        if full_key not in self.editor.entry_map:
            await interaction.response.send_message(
                "Selected value no longer exists.", ephemeral=True
            )
            return
        current_value = self.editor.entry_map[full_key]
        await interaction.response.send_modal(
            ConfigValueModal(self.editor, full_key, current_value)
        )


class ConfigEditorView(View):
    def __init__(self, bot: "BacktestBot", entries: List[tuple[str, Any]]) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.entries = entries
        self.entry_map: Dict[str, Any] = {key: value for key, value in entries}
        self.page = 0
        self.page_size = 25
        self.message: Optional[discord.Message] = None

        self.select = ConfigSelect(self)
        self.prev_button = Button(label="Prev", style=discord.ButtonStyle.secondary)
        self.next_button = Button(label="Next", style=discord.ButtonStyle.secondary)
        self.prev_button.callback = self.prev_page
        self.next_button.callback = self.next_page

        self.add_item(self.select)
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self._update_buttons()

    def _update_buttons(self) -> None:
        max_pages = max(1, (len(self.entries) - 1) // self.page_size + 1)
        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= max_pages - 1

    def _page_entries(self) -> List[tuple[str, Any]]:
        start = self.page * self.page_size
        end = start + self.page_size
        return self.entries[start:end]

    def options_for_page(self) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = []
        for full_key, value in self._page_entries():
            display_value = self._format_value(value)
            options.append(
                discord.SelectOption(
                    label=full_key[:100],
                    description=display_value[:100],
                    value=full_key,
                )
            )
        if not options:
            options.append(
                discord.SelectOption(label="No editable values", value="__none__")
            )
        return options

    def _format_value(self, value: Any) -> str:
        text = str(value)
        if len(text) > 100:
            return text[:97] + "..."
        return text

    def _page_summary(self) -> str:
        max_pages = max(1, (len(self.entries) - 1) // self.page_size + 1)
        lines = [f"Page {self.page + 1}/{max_pages}"]
        max_len = 1800
        for full_key, value in self._page_entries():
            line = f"- {full_key}: {self._format_value(value)}"
            if len("\n".join(lines + [line])) > max_len:
                lines.append("- ...")
                break
            lines.append(line)
        return "\n".join(lines)

    async def prev_page(self, interaction: discord.Interaction) -> None:
        if self.page > 0:
            self.page -= 1
        await self.refresh_message(interaction)

    async def next_page(self, interaction: discord.Interaction) -> None:
        max_pages = max(1, (len(self.entries) - 1) // self.page_size + 1)
        if self.page < max_pages - 1:
            self.page += 1
        await self.refresh_message(interaction)

    def apply_change(self, full_key: str, raw_value: str) -> bool:
        if full_key not in self.entry_map:
            return False
        if "." not in full_key:
            return False
        section, key = full_key.split(".", 1)
        coerced = self.bot._coerce_value(raw_value)
        updated = False
        if "[" in key and key.endswith("]"):
            base_key, index_part = key.rsplit("[", 1)
            index_text = index_part[:-1]
            if index_text.isdigit():
                list_index = int(index_text) - 1
                current_list = self.bot._get_setting_value(section, base_key)
                if isinstance(current_list, list) and 0 <= list_index < len(current_list):
                    current_list[list_index] = coerced
                    self.bot._apply_setting(section, base_key, current_list)
                    updated = True
        if not updated:
            self.bot._apply_setting(section, key, coerced)
            updated = True
        if not updated:
            return False
        save_settings(self.bot.settings_path, self.bot.settings)
        self.entries = self.bot._flatten_settings()
        self.entry_map = {entry_key: value for entry_key, value in self.entries}
        return True

    async def refresh_message(self, interaction: discord.Interaction) -> None:
        self.select.options = self.options_for_page()
        self._update_buttons()
        content = "Select a variable to edit.\n" + self._page_summary()
        if interaction.response.is_done():
            if interaction.message:
                await interaction.message.edit(content=content, view=self)
            elif self.message:
                await self.message.edit(content=content, view=self)
            return
        if interaction.message:
            await interaction.response.edit_message(content=content, view=self)
            return
        await interaction.response.defer()
        if self.message:
            await self.message.edit(content=content, view=self)


class TextPagerView(View):
    def __init__(
        self,
        title: str,
        lines: List[str],
        *,
        page_size: int = 25,
        max_len: int = 1800,
        code_block: Optional[str] = None,
    ) -> None:
        super().__init__(timeout=300)
        self.title = title
        self.lines = lines
        self.page = 0
        self.page_size = page_size
        self.max_len = max_len
        self.code_block = code_block
        self.message: Optional[discord.Message] = None

        self.prev_button = Button(label="Prev", style=discord.ButtonStyle.secondary)
        self.next_button = Button(label="Next", style=discord.ButtonStyle.secondary)
        self.prev_button.callback = self.prev_page
        self.next_button.callback = self.next_page
        self.add_item(self.prev_button)
        self.add_item(self.next_button)
        self._update_buttons()

    def _max_pages(self) -> int:
        return max(1, (len(self.lines) - 1) // self.page_size + 1)

    def _update_buttons(self) -> None:
        max_pages = self._max_pages()
        self.prev_button.disabled = self.page <= 0
        self.next_button.disabled = self.page >= max_pages - 1

    def _page_lines(self) -> List[str]:
        start = self.page * self.page_size
        end = start + self.page_size
        return self.lines[start:end]

    def _render_plain(self) -> str:
        max_pages = self._max_pages()
        header = f"{self.title} (Page {self.page + 1}/{max_pages})"
        if not self.lines:
            return f"{header}\n- (no entries)"
        lines = [header]
        for line in self._page_lines():
            if len("\n".join(lines + [line])) > self.max_len:
                lines.append("...")
                break
            lines.append(line)
        return "\n".join(lines)

    def _render_code(self) -> str:
        max_pages = self._max_pages()
        header = f"{self.title} (Page {self.page + 1}/{max_pages})"
        prefix = f"{header}\n```{self.code_block or ''}\n"
        suffix = "\n```"
        if not self.lines:
            return prefix + "# (no entries)" + suffix
        body: List[str] = []
        for line in self._page_lines():
            candidate = prefix + "\n".join(body + [line]) + suffix
            if len(candidate) > self.max_len:
                if not body:
                    available = self.max_len - len(prefix) - len(suffix) - 3
                    body.append(line[: max(0, available)] + "...")
                else:
                    body.append("...")
                break
            body.append(line)
        return prefix + "\n".join(body) + suffix

    def page_content(self) -> str:
        if self.code_block is not None:
            return self._render_code()
        return self._render_plain()

    async def refresh_message(self, interaction: discord.Interaction) -> None:
        self._update_buttons()
        content = self.page_content()
        if interaction.message:
            await interaction.response.edit_message(content=content, view=self)
            return
        await interaction.response.defer()
        if self.message:
            await self.message.edit(content=content, view=self)

    async def prev_page(self, interaction: discord.Interaction) -> None:
        if self.page > 0:
            self.page -= 1
        await self.refresh_message(interaction)

    async def next_page(self, interaction: discord.Interaction) -> None:
        max_pages = self._max_pages()
        if self.page < max_pages - 1:
            self.page += 1
        await self.refresh_message(interaction)

    async def on_timeout(self) -> None:
        self.prev_button.disabled = True
        self.next_button.disabled = True
        if self.message:
            await self.message.edit(view=self)


class BacktestBot(commands.Bot):
    def __init__(self, settings: Settings, settings_path: Path) -> None:
        discord_cfg = settings.discord
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix=discord_cfg.get("command_prefix", "!"),
            intents=intents,
            help_command=None,
        )
        self.settings = settings
        self.settings_path = settings_path
        self.command_channel_name = discord_cfg.get("command_channel", "backtest-commands")
        self.result_channel_name = discord_cfg.get("result_channel", "backtest-results")
        self.data_list_channel_name = discord_cfg.get("data_list_channel", "backtest-data")
        self.strategy_list_channel_name = discord_cfg.get(
            "strategy_list_channel", "backtest-strategies"
        )
        self.main_category_name = discord_cfg.get("main_category", "main")
        self.chat_channel_name = discord_cfg.get("chat_channel", "chat")
        self.backtest_category_name = discord_cfg.get("backtest_category", "backtest")
        self.sim_category_name = discord_cfg.get("sim_category", "Sim")
        self.live_category_name = discord_cfg.get("live_category", "Live")
        self.sim_command_channel_name = discord_cfg.get("sim_command_channel", "sim-commands")
        self.sim_status_channel_name = discord_cfg.get("sim_status_channel", "sim-status")
        self.sim_result_channel_name = discord_cfg.get("sim_result_channel", "sim-results")
        self.sim_tradelog_channel_name = discord_cfg.get("sim_tradelog_channel", "sim-tradelog")
        self.live_command_channel_name = discord_cfg.get("live_command_channel", "live-commands")
        self.live_status_channel_name = discord_cfg.get("live_status_channel", "live-status")
        self.live_result_channel_name = discord_cfg.get("live_result_channel", "live-results")
        self.live_tradelog_channel_name = discord_cfg.get("live_tradelog_channel", "live-tradelog")
        self.sim_channel_names = list(discord_cfg.get("sim_channels", []))
        self.live_channel_names = list(discord_cfg.get("live_channels", []))
        self.backtest_channel_names = [
            name
            for name in (
                self.command_channel_name,
                self.result_channel_name,
                self.data_list_channel_name,
                self.strategy_list_channel_name,
            )
            if name
        ]
        self.command_channel_names = [
            name
            for name in (
                self.command_channel_name,
                self.sim_command_channel_name,
                self.live_command_channel_name,
            )
            if name
        ]
        extra_backtest_channels = discord_cfg.get("backtest_channels", [])
        for name in extra_backtest_channels:
            if name and name not in self.backtest_channel_names:
                self.backtest_channel_names.append(name)
        for name in (
            self.sim_command_channel_name,
            self.sim_status_channel_name,
            self.sim_result_channel_name,
            self.sim_tradelog_channel_name,
        ):
            if name and name not in self.sim_channel_names:
                self.sim_channel_names.append(name)
        for name in (
            self.live_command_channel_name,
            self.live_status_channel_name,
            self.live_result_channel_name,
            self.live_tradelog_channel_name,
        ):
            if name and name not in self.live_channel_names:
                self.live_channel_names.append(name)
        self.create_channels = bool(discord_cfg.get("create_channels", True))
        self.enable_slash = bool(discord_cfg.get("enable_slash", False))
        self.guild_id = discord_cfg.get("guild_id")
        self.setting_profiles_dir = Path(
            discord_cfg.get("setting_profiles_dir", "config/setting_profiles")
        )
        self.status_sources = dict(discord_cfg.get("status_sources", {}))
        self.status_interval_minutes = int(
            discord_cfg.get("status_interval_minutes", 60)
        )
        self.sim_cfg = discord_cfg.get("sim", {}) if isinstance(discord_cfg.get("sim", {}), dict) else {}
        self.live_cfg = discord_cfg.get("live", {}) if isinstance(discord_cfg.get("live", {}), dict) else {}
        self.sim_enabled = bool(self.sim_cfg.get("enabled", False))
        self.live_enabled = bool(self.live_cfg.get("enabled", False))
        self._realtime_payloads: Dict[str, RunnerPayload] = {}
        self.sim_runner: Optional[RealtimeRunner] = None
        self.live_runner: Optional[RealtimeRunner] = None
        self._status_task: Optional[asyncio.Task] = None
        self._post_ready_sync_task: Optional[asyncio.Task] = None
        self._sync_lock = asyncio.Lock()
        self._last_sync_at = 0.0
        self._min_sync_interval = 8.0
        self._last_backtest: Optional[Dict[str, Any]] = None
        self._active_grid_runs: Dict[str, GridProgressHandle] = {}
        self._pending_grid_outputs: List[PendingGridOutput] = []
        self._wordle_games: Dict[int, WordleGame] = {}

        # Register commands per-guild during sync to avoid stale global duplicates.
        if self.enable_slash:
            self._guild_commands = [
                self.backtest_slash,
                self.download_data_slash,
                self.set_config_slash,
                self.set_config_bulk_slash,
                self.config_edit_slash,
                self.sync_slash,
            ]
            # Ensure app commands are bound to this bot instance (for methods with self).
            for command in self._guild_commands:
                command.binding = self
            # Bind autocompletes after command binding so "self" is present.
            self.backtest_slash.autocomplete("data_file")(self.backtest_data_autocomplete)
            self.backtest_slash.autocomplete("strategy")(self.backtest_strategy_autocomplete)
        else:
            self._guild_commands = []

        self._text_commands = [
            commands.Command(sync_text_command, name="sync"),
            commands.Command(download_text_command, name="download"),
            commands.Command(delete_data_text_command, name="delete_data"),
            commands.Command(backtest_text_command, name="testrun"),
            commands.Command(entry_diff_text_command, name="diff"),
            commands.Command(grid_text_command, name="grid"),
            commands.Command(comp_text_command, name="comp"),
            commands.Command(costom_text_command, name="costom"),
            commands.Command(costom2_text_command, name="costom2"),
            commands.Command(costom3_text_command, name="costom3"),
            commands.Command(costom4_text_command, name="costom4"),
            commands.Command(chart_text_command, name="chart"),
            commands.Command(graph_text_command, name="graph"),
            commands.Command(stge_text_command, name="stge"),
            commands.Command(set_config_text_command, name="set_config"),
            commands.Command(set_config_bulk_text_command, name="set_config_bulk"),
            commands.Command(config_edit_text_command, name="conf"),
            commands.Command(status_text_command, name="status"),
            commands.Command(setting_text_command, name="setting"),
            commands.Command(sim_text_command, name="sim"),
            commands.Command(live_text_command, name="live"),
            commands.Command(order_text_command, name="order"),
            commands.Command(close_text_command, name="close"),
            commands.Command(positions_text_command, name="positions"),
            commands.Command(assets_text_command, name="assets"),
            commands.Command(wordle_text_command, name="wordle"),
            commands.Command(help_text_command, name="help"),
        ]
        for command in self._text_commands:
            self.add_command(command)

    @staticmethod
    def _split_message(message: str, limit: int = 2000) -> List[str]:
        if not message:
            return [""]
        lines = message.splitlines(keepends=True)
        chunks: List[str] = []
        current = ""
        for line in lines:
            if len(line) > limit:
                if current:
                    chunks.append(current.rstrip("\n"))
                    current = ""
                start = 0
                while start < len(line):
                    part = line[start : start + limit]
                    chunks.append(part.rstrip("\n"))
                    start += limit
                continue
            if len(current) + len(line) > limit:
                chunks.append(current.rstrip("\n"))
                current = ""
            current += line
        if current or not chunks:
            chunks.append(current.rstrip("\n"))
        return chunks

    async def on_ready(self) -> None:
        await self._log_connected_guilds()
        await self._ensure_channels()
        await self._ensure_realtime_runners()
        await self._update_data_list_channel()
        await self._update_strategy_list_channel()
        await self._reissue_grid_progress()
        await self._flush_pending_grid_outputs()
        self._start_status_task()
        if self.enable_slash:
            await self._sync_commands()
            if self._post_ready_sync_task is None or self._post_ready_sync_task.done():
                self._post_ready_sync_task = asyncio.create_task(self._delayed_resync())

    async def on_resumed(self) -> None:
        await self._reissue_grid_progress()
        await self._flush_pending_grid_outputs()

    async def on_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        root_error: Exception = error
        if isinstance(error, app_commands.CommandInvokeError) and error.original:
            root_error = error.original
        if isinstance(root_error, (app_commands.CommandSignatureMismatch, app_commands.CommandNotFound)):
            if interaction.guild is not None:
                try:
                    await self._sync_commands(guild=interaction.guild)
                except Exception:
                    pass
            message = (
                "Commands were updated; I resynced this server. "
                "Please retry the slash command in a few seconds."
            )
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
            return
        raise error

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        raise error

    def _register_grid_progress(
        self,
        message: Optional[discord.Message],
        channel_id: int,
        total: int,
    ) -> GridProgressHandle:
        run_id = uuid.uuid4().hex
        handle = GridProgressHandle(run_id, channel_id, total, message=message)
        self._active_grid_runs[run_id] = handle
        return handle

    def _unregister_grid_progress(self, handle: GridProgressHandle) -> None:
        self._active_grid_runs.pop(handle.run_id, None)

    async def _reissue_grid_progress(self) -> None:
        if not self._active_grid_runs:
            return
        for handle in list(self._active_grid_runs.values()):
            channel = self.get_channel(handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(handle.channel_id)
                except Exception as exc:
                    logger.warning("Failed to fetch grid progress channel: %s", exc)
                    continue
            pct = int((handle.done / handle.total) * 100) if handle.total else 100
            try:
                handle.message = await channel.send(
                    f"Running grid backtests ({handle.done}/{handle.total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Failed to reissue grid progress message: %s", exc)
    
    async def _send_grid_output_to_channel(
        self,
        channel_id: int,
        content: str,
        image_paths: Optional[Sequence[Path]] = None,
        data_paths: Optional[Sequence[Path]] = None,
    ) -> bool:
        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except Exception as exc:
                logger.warning("Failed to fetch grid output channel: %s", exc)
                return False
        try:
            await self._send_grid_output(
                channel.send,
                content,
                image_paths=image_paths,
                data_paths=data_paths,
            )
            return True
        except discord.HTTPException as exc:
            logger.warning("Failed to send grid output: %s", exc)
            return False

    def _queue_grid_output(
        self,
        channel_id: int,
        content: str,
        image_paths: Optional[Sequence[Path]] = None,
        data_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        self._pending_grid_outputs.append(
            PendingGridOutput(channel_id, content, image_paths, data_paths)
        )

    async def _flush_pending_grid_outputs(self) -> None:
        if not self._pending_grid_outputs:
            return
        pending = list(self._pending_grid_outputs)
        self._pending_grid_outputs.clear()
        for item in pending:
            sent = await self._send_grid_output_to_channel(
                item.channel_id,
                item.content,
                image_paths=item.image_paths,
                data_paths=item.data_paths,
            )
            if not sent:
                self._pending_grid_outputs.append(item)

    def _application_id(self) -> Optional[int]:
        if self.application_id:
            return int(self.application_id)
        if self.user:
            return int(self.user.id)
        return None

    async def _http_clear_commands(
        self, guild: Optional[discord.abc.Snowflake]
    ) -> None:
        app_id = self._application_id()
        if app_id is None:
            return
        if not hasattr(self.http, "bulk_upsert_global_commands"):
            return
        if guild is None:
            await self.http.bulk_upsert_global_commands(app_id, [])
            return
        await self.http.bulk_upsert_guild_commands(app_id, int(guild.id), [])

    async def _purge_commands_on_discord(
        self, guild: Optional[discord.abc.Snowflake]
    ) -> None:
        # Clear remote commands only; keep local tree for dispatch.
        await self._http_clear_commands(guild=guild)

    def _register_commands_locally(self) -> None:
        # Keep commands as local globals; sync(guild=...) routes them to guilds.
        self.tree.clear_commands(guild=None)
        for command in self._guild_commands:
            self.tree.add_command(command)

    async def _sync_guild_only(self, guild: discord.abc.Snowflake) -> None:
        # Hard reset guild commands: sync empty list first, then register.
        self.tree.clear_commands(guild=guild)
        try:
            await self._safe_tree_sync(guild=guild)
        except Exception:
            pass
        await self._purge_commands_on_discord(guild=guild)
        self._register_commands_locally()
        await self._safe_tree_sync(guild=guild)

    async def _safe_tree_sync(
        self, guild: Optional[discord.abc.Snowflake] = None
    ) -> None:
        try:
            await self.tree.sync(guild=guild)
        except discord.HTTPException as exc:
            if exc.status != 429:
                raise
            retry_after = getattr(exc, "retry_after", None)
            await asyncio.sleep(retry_after or 2.0)
            await self.tree.sync(guild=guild)

    async def _sync_commands(self, guild: Optional[discord.abc.Snowflake] = None) -> None:
        async with self._sync_lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - self._last_sync_at
            if elapsed < self._min_sync_interval:
                await asyncio.sleep(self._min_sync_interval - elapsed)
            if guild is not None:
                await self._purge_commands_on_discord(guild=None)
                await self._sync_guild_only(guild)
                self._last_sync_at = asyncio.get_running_loop().time()
                return
            if self.guild_id:
                guild_obj = discord.Object(id=int(self.guild_id))
                await self._purge_commands_on_discord(guild=None)
                await self._sync_guild_only(guild_obj)
                self._last_sync_at = asyncio.get_running_loop().time()
                return
            guilds = await self._resolve_guilds()
            if guilds:
                await self._purge_commands_on_discord(guild=None)
                for index, guild_obj in enumerate(guilds):
                    await self._sync_guild_only(guild_obj)
                    if index < len(guilds) - 1:
                        await asyncio.sleep(2.0)
                self._last_sync_at = asyncio.get_running_loop().time()
                return
            # No guild context available; fall back to global sync.
            self._register_commands_locally()
            await self._safe_tree_sync(guild=None)
            self._last_sync_at = asyncio.get_running_loop().time()

    async def _resolve_guilds(self) -> List[discord.abc.Snowflake]:
        cached = list(self.guilds)
        if cached:
            return cached
        guilds: List[discord.abc.Snowflake] = []
        try:
            async for guild in self.fetch_guilds(limit=None):
                guilds.append(discord.Object(id=guild.id))
        except Exception:
            return []
        return guilds

    async def _log_connected_guilds(self) -> None:
        guilds: List[discord.Guild] = list(self.guilds)
        if not guilds:
            try:
                async for guild in self.fetch_guilds(limit=None):
                    guilds.append(guild)
            except Exception as exc:
                logger.warning("Failed to fetch guild list: %s", exc)
                return
        if not guilds:
            logger.info("Connected guilds: none")
            return
        names = ", ".join(f"{guild.name}({guild.id})" for guild in guilds)
        logger.info("Connected guilds (%d): %s", len(guilds), names)

    async def _delayed_resync(self) -> None:
        await asyncio.sleep(10)
        try:
            if asyncio.get_running_loop().time() - self._last_sync_at < self._min_sync_interval:
                return
            await self._sync_commands()
        except Exception:
            return

    def _start_status_task(self) -> None:
        if self._status_task is None or self._status_task.done():
            self._status_task = asyncio.create_task(self._status_loop())

    def _seconds_until_next_status_slot(self, interval_minutes: int) -> float:
        interval = max(int(interval_minutes), 1)
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        base = now.replace(second=0, microsecond=0)
        next_minute = (base.minute // interval) * interval + interval
        if next_minute >= 60:
            target = base.replace(minute=0) + timedelta(hours=1)
        else:
            target = base.replace(minute=next_minute)
        delta = (target - now).total_seconds()
        if delta < 0.5:
            return interval * 60
        return delta

    async def _status_loop(self) -> None:
        interval = max(self.status_interval_minutes, 1)
        await asyncio.sleep(self._seconds_until_next_status_slot(interval))
        while not self.is_closed():
            try:
                await self._post_status_updates()
            except Exception as exc:
                logger.warning("Failed to post status update: %s", exc)
            await asyncio.sleep(self._seconds_until_next_status_slot(interval))

    def _status_source_for_group(self, group: str) -> Optional[Any]:
        if not isinstance(self.status_sources, dict):
            return None
        source = self.status_sources.get(group)
        if source is not None:
            return source
        channel_name = self._group_channel_name(group, "status")
        if channel_name:
            return self.status_sources.get(channel_name)
        return None

    def _runner_for_group(self, group: str) -> Optional[RealtimeRunner]:
        if group == "sim":
            return self.sim_runner
        if group == "live":
            return self.live_runner
        return None

    def _payload_to_snapshot(self, payload: RunnerPayload) -> Optional[StatusSnapshot]:
        if payload is None:
            return None
        payload_dict = {
            "symbol": payload.symbol,
            "interval": payload.interval,
            "price": payload.price,
            "equity": payload.equity,
            "free_balance": payload.free_balance,
            "positions": payload.positions,
            "updated_at": payload.updated_at,
        }
        return self._parse_status_snapshot(payload_dict)

    def _infer_symbol_from_pattern(self, pattern: str) -> str:
        if not pattern:
            return "UNKNOWN"
        name = pattern.split("/")[-1]
        stem = name.split(".")[0]
        parts = stem.split("_")
        if len(parts) >= 4:
            return "_".join(parts[:-3])
        return stem.upper() if stem else "UNKNOWN"

    def _infer_interval_from_pattern(self, pattern: str) -> str:
        if not pattern:
            return ""
        name = pattern.split("/")[-1]
        stem = name.split(".")[0]
        parts = stem.split("_")
        if len(parts) >= 4:
            return parts[-3]
        return ""

    def _build_position_payloads_from_open(
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

            tp_pnl = RealtimeRunner._target_pnl(entry_price, tp_price, direction, leverage)
            sl_pnl = RealtimeRunner._target_pnl(entry_price, sl_price, direction, leverage)

            payloads[side] = {
                "active": True,
                "qty": qty,
                "notional": notional,
                "leverage": leverage,
                "entry_price": entry_price,
                "direction": direction,
                "tp_pnl": tp_pnl,
                "sl_pnl": sl_pnl,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "pnl_pct": pnl_pct,
                "pnl_amount": pnl_amount,
            }
        return payloads

    def _backtest_snapshot(self) -> Optional[StatusSnapshot]:
        if self._last_backtest is None:
            return None
        result = self._last_backtest.get("result")
        data = self._last_backtest.get("data")
        if result is None or data is None or data.empty:
            return None
        last_row = data.iloc[-1]
        price = float(last_row.get("close", 0.0) or 0.0)
        equity = (
            float(result.equity_curve["equity"].iloc[-1])
            if not result.equity_curve.empty
            else 0.0
        )
        free_balance = float(result.final_cash)
        positions = self._build_position_payloads_from_open(
            result.open_positions or {}, price
        )
        symbol = self._infer_symbol_from_pattern(
            str(self.settings.data.get("file_pattern", ""))
        )
        interval = self._infer_interval_from_pattern(
            str(self.settings.data.get("file_pattern", ""))
        )
        payload = {
            "symbol": symbol,
            "interval": interval,
            "price": price,
            "equity": equity,
            "free_balance": free_balance,
            "positions": positions,
            "updated_at": time.time(),
        }
        return self._parse_status_snapshot(payload)

    def _status_snapshot_for_group(self, group: str) -> Optional[StatusSnapshot]:
        if group in {"sim", "live"}:
            payload = self._realtime_payloads.get(group)
            snapshot = self._payload_to_snapshot(payload) if payload else None
            if snapshot is not None:
                if not snapshot.interval:
                    cfg = self._get_realtime_cfg(group)
                    snapshot.interval = str(cfg.get("interval", "") or "")
                return snapshot
        if group == "backtest":
            snapshot = self._backtest_snapshot()
            if snapshot is not None:
                return snapshot
        source = self._status_source_for_group(group)
        if source is None:
            return None
        path = None
        if isinstance(source, dict):
            raw_path = source.get("path") or source.get("file")
        else:
            raw_path = source
        if raw_path:
            path = Path(str(raw_path))
        if path is None:
            return None
        payload = self._read_status_payload(path)
        if payload is None:
            return None
        return self._parse_status_snapshot(payload)

    @staticmethod
    def _read_status_payload(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return None
        if not raw.strip():
            return None
        try:
            if path.suffix.lower() in {".json"}:
                return json.loads(raw)
            return yaml.safe_load(raw)
        except Exception:
            return None

    @staticmethod
    def _parse_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_target_pnl(
        self,
        entry_price: float,
        target_price: Any,
        direction: int,
        leverage: float,
        raw_pnl: Any,
    ) -> Optional[float]:
        if raw_pnl is not None:
            return self._parse_float(raw_pnl)
        if entry_price <= 0 or direction == 0:
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

    def _parse_position_status(
        self,
        payload: Dict[str, Any],
        side: str,
        price: float,
    ) -> PositionStatus:
        positions = payload.get("positions")
        if isinstance(positions, dict):
            data = positions.get(side, {})
        else:
            data = payload.get(side, {})
        if not isinstance(data, dict):
            data = {}

        qty = self._parse_float(data.get("qty", 0.0))
        notional = self._parse_float(data.get("notional", 0.0))
        if notional == 0.0 and qty and price:
            notional = qty * price
        leverage = self._parse_float(data.get("leverage", data.get("lev", 1.0)), 1.0)
        entry_price = self._parse_float(data.get("entry_price", 0.0))
        direction = int(self._parse_float(data.get("direction", 0.0)))
        tp_pnl = data.get("tp_pnl", data.get("tp", data.get("tp_pct")))
        sl_pnl = data.get("sl_pnl", data.get("sl", data.get("sl_pct")))
        tp_price = data.get("tp_price", data.get("rr_tp_price"))
        sl_price = data.get("sl_price", data.get("rr_stop_price", data.get("stop_price")))
        pnl_pct = self._parse_float(
            data.get("pnl_pct", data.get("pnl_percent", data.get("return_pct", 0.0)))
        )
        pnl_amount = self._parse_float(
            data.get("pnl_amount", data.get("pnl_usdt", data.get("pnl_value", 0.0)))
        )
        active = bool(
            data.get("active", data.get("open", data.get("in_position", False)))
            or qty > 0
            or notional > 0
        )
        return PositionStatus(
            side=side,
            active=active,
            notional=notional,
            qty=qty,
            leverage=leverage if leverage > 0 else 1.0,
            tp_pnl=self._resolve_target_pnl(entry_price, tp_price, direction, leverage, tp_pnl),
            sl_pnl=self._resolve_target_pnl(entry_price, sl_price, direction, leverage, sl_pnl),
            tp_price=self._parse_float(tp_price) if tp_price is not None else None,
            sl_price=self._parse_float(sl_price) if sl_price is not None else None,
            pnl_pct=pnl_pct,
            pnl_amount=pnl_amount,
        )

    def _parse_status_snapshot(self, payload: Dict[str, Any]) -> Optional[StatusSnapshot]:
        if not isinstance(payload, dict):
            return None
        symbol = str(payload.get("symbol", payload.get("ticker", ""))).strip() or "UNKNOWN"
        interval = str(payload.get("interval", payload.get("timeframe", payload.get("bar", ""))) or "").strip()
        price = self._parse_float(payload.get("price", payload.get("last_price", 0.0)))
        equity = self._parse_float(payload.get("equity", payload.get("total_equity", 0.0)))
        free_balance = self._parse_float(
            payload.get("free_balance", payload.get("available_balance", payload.get("free", 0.0)))
        )
        updated_at = payload.get("updated_at", payload.get("timestamp"))
        updated_ts = 0.0
        if isinstance(updated_at, (int, float)):
            updated_ts = float(updated_at)
        elif isinstance(updated_at, str) and updated_at:
            try:
                updated_ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                updated_ts = 0.0

        long_status = self._parse_position_status(payload, "long", price)
        short_status = self._parse_position_status(payload, "short", price)

        return StatusSnapshot(
            symbol=symbol,
            interval=interval,
            price=price,
            equity=equity,
            free_balance=free_balance,
            long=long_status,
            short=short_status,
            updated_at=updated_ts,
        )

    def _runner_status_text(self, group: str) -> str:
        if group == "sim":
            runner = self.sim_runner
            enabled = self.sim_enabled
        elif group == "live":
            runner = self.live_runner
            enabled = self.live_enabled
        else:
            return "N/A"

        if runner is None:
            return "STOPPED" if enabled else "DISABLED"
        return "RUNNING" if runner.is_running() else "STOPPED"

    @staticmethod
    def _format_kst_time(value: Any) -> str:
        if value is None or value == "":
            return "n/a"
        dt: Optional[datetime] = None
        if isinstance(value, (int, float)):
            ts_val = float(value)
            if np.isfinite(ts_val):
                abs_val = abs(ts_val)
                if abs_val >= 1e18:
                    ts_val = ts_val / 1e9
                elif abs_val >= 1e15:
                    ts_val = ts_val / 1e6
                elif abs_val >= 1e12:
                    ts_val = ts_val / 1e3
                dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
        elif isinstance(value, pd.Timestamp):
            ts = value
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(ZoneInfo("Asia/Seoul"))
            dt = ts.to_pydatetime()
        elif isinstance(value, datetime):
            dt = value
        else:
            ts_val = None
            try:
                ts_val = float(value)
            except (TypeError, ValueError):
                ts_val = None
            if ts_val is not None and np.isfinite(ts_val):
                abs_val = abs(ts_val)
                if abs_val >= 1e18:
                    ts_val = ts_val / 1e9
                elif abs_val >= 1e15:
                    ts_val = ts_val / 1e6
                elif abs_val >= 1e12:
                    ts_val = ts_val / 1e3
                dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            else:
                ts = pd.to_datetime(value, utc=True, errors="coerce")
                if ts is not pd.NaT:
                    ts = ts.tz_convert(ZoneInfo("Asia/Seoul"))
                    dt = ts.to_pydatetime()
        if dt is None:
            return "n/a"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))
        else:
            dt = dt.astimezone(ZoneInfo("Asia/Seoul"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _fmt_price(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value)
            return text if text else "n/a"
        text = f"{number:,.6f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _format_status_embed(
        self, snapshot: StatusSnapshot, title: str, group: str
    ) -> discord.Embed:
        def fmt_money(value: float) -> str:
            return f"{value:,.2f}"

        def fmt_pct(value: float) -> str:
            return f"{value:+.2f}%"

        def fmt_optional(value: Optional[float]) -> str:
            if value is None:
                return "n/a"
            return f"{value:+.2f}%"

        def fmt_roi(initial_balance: float, equity: float) -> str:
            if initial_balance <= 0:
                return "n/a"
            return fmt_pct((equity / initial_balance - 1.0) * 100.0)

        def fmt_target(pnl_value: Optional[float], price_value: Optional[float]) -> str:
            if pnl_value is None and price_value is None:
                return "n/a"
            pnl_text = fmt_optional(pnl_value) if pnl_value is not None else "n/a"
            if price_value is None:
                return pnl_text
            return f"{pnl_text} ({self._fmt_price(price_value)})"

        updated = ""
        if snapshot.updated_at > 0:
            updated = datetime.fromtimestamp(
                snapshot.updated_at, tz=ZoneInfo("Asia/Seoul")
            ).strftime("%Y-%m-%d %H:%M:%S")

        status_text = self._runner_status_text(group)
        status_color = 0x2ECC71 if status_text == "RUNNING" else 0xE67E22
        if status_text == "DISABLED":
            status_color = 0x95A5A6
        if group == "backtest":
            status_color = 0x3498DB

        embed = discord.Embed(title=f"📡 {title} 상태", color=status_color)
        interval_text = snapshot.interval or "n/a"
        embed.add_field(
            name="시세",
            value=(
                f"심볼 `{snapshot.symbol}`\n"
                f"인터벌 `{interval_text}`\n"
                f"현재가 {self._fmt_price(snapshot.price)}"
            ),
            inline=True,
        )
        if group == "sim":
            initial_balance = self._sim_initial_balance()
            roi_text = fmt_roi(initial_balance, snapshot.equity)
            asset_value = (
                f"설정 {fmt_money(initial_balance)}\n"
                f"초기자산 대비 {roi_text}\n"
                f"총 {fmt_money(snapshot.equity)}\n"
                f"가용 {fmt_money(snapshot.free_balance)}"
            )
        else:
            asset_value = (
                f"총 {fmt_money(snapshot.equity)}\n"
                f"가용 {fmt_money(snapshot.free_balance)}"
            )
        embed.add_field(name="자산", value=asset_value, inline=True)
        embed.add_field(name="봇 상태", value=status_text, inline=True)

        long = snapshot.long
        long_value = (
            f"{'보유중' if long.active else '미보유'}\n"
            f"수량 {long.qty:.6g} | 금액 {fmt_money(long.notional)}\n"
            f"레버리지 {long.leverage:.2f}x\n"
            f"TP {fmt_target(long.tp_pnl, long.tp_price)} | SL {fmt_target(long.sl_pnl, long.sl_price)}\n"
            f"수익률 {fmt_pct(long.pnl_pct)} | 수익금 {fmt_money(long.pnl_amount)}"
        )
        embed.add_field(name="🟢 롱", value=long_value, inline=False)

        short = snapshot.short
        short_value = (
            f"{'보유중' if short.active else '미보유'}\n"
            f"수량 {short.qty:.6g} | 금액 {fmt_money(short.notional)}\n"
            f"레버리지 {short.leverage:.2f}x\n"
            f"TP {fmt_target(short.tp_pnl, short.tp_price)} | SL {fmt_target(short.sl_pnl, short.sl_price)}\n"
            f"수익률 {fmt_pct(short.pnl_pct)} | 수익금 {fmt_money(short.pnl_amount)}"
        )
        embed.add_field(name="🔴 숏", value=short_value, inline=False)

        if updated:
            embed.set_footer(text=f"업데이트: {updated} KST")
        return embed

    async def _post_status_updates(self) -> None:
        groups = ("sim", "live")
        for group in groups:
            snapshot = self._status_snapshot_for_group(group)
            if snapshot is None:
                continue
            channel = await self._get_or_create_group_channel(
                group, self._group_channel_name(group, "status")
            )
            if channel is None:
                continue
            title = "Backtest" if group == "backtest" else group.capitalize()
            embed = self._format_status_embed(snapshot, title=title, group=group)
            try:
                await self._clear_status_channel(channel)
                await channel.send(embed=embed)
            except discord.HTTPException:
                continue

    async def _clear_status_channel(self, channel: discord.TextChannel) -> None:
        try:
            while True:
                messages = [m async for m in channel.history(limit=100)]
                if not messages:
                    break
                try:
                    await channel.delete_messages(messages)
                except discord.HTTPException:
                    for msg in messages:
                        try:
                            await msg.delete()
                        except discord.HTTPException:
                            continue
        except (discord.Forbidden, discord.HTTPException):
            return

    async def _ensure_realtime_runners(self) -> None:
        if self.sim_enabled and self.sim_runner is None:
            runner = self._build_realtime_runner("sim")
            if runner is not None:
                self.sim_runner = runner
                runner.start()
        if self.live_enabled and self.live_runner is None:
            runner = self._build_realtime_runner("live")
            if runner is not None:
                self.live_runner = runner
                runner.start()

    def _build_realtime_runner(self, group: str) -> Optional[RealtimeRunner]:
        cfg = self.sim_cfg if group == "sim" else self.live_cfg
        if not isinstance(cfg, dict):
            cfg = {}
        def _as_float(value: Any, default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _as_int(value: Any, default: int) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        symbol = str(cfg.get("symbol", "XRPUSDT")).upper()
        interval = str(cfg.get("interval", "5m"))
        initial_balance = _as_float(
            cfg.get("initial_balance", self.settings.backtest.get("initial_balance")), 0.0
        )
        fee_rate = _as_float(
            cfg.get("fee_rate", self.settings.backtest.get("fee_rate")), 0.0
        )
        leverage = _as_float(
            cfg.get("leverage", self.settings.backtest.get("leverage")), 1.0
        )
        position_size = _as_float(
            cfg.get("position_size", self.settings.backtest.get("position_size")), 1.0
        )
        slippage = _as_float(
            cfg.get("slippage", self.settings.backtest.get("slippage")), 0.0
        )
        risk_per_trade = cfg.get("risk_per_trade", self.settings.backtest.get("risk_per_trade"))
        if risk_per_trade is not None:
            risk_per_trade = _as_float(risk_per_trade, 0.0)
        max_pos_fraction = cfg.get(
            "max_position_fraction_per_side",
            self.settings.backtest.get("max_position_fraction_per_side"),
        )
        if max_pos_fraction is not None:
            max_pos_fraction = _as_float(max_pos_fraction, 0.0)
        poll_interval = _as_float(cfg.get("poll_interval", 10.0), 10.0)
        signal_delay_seconds = _as_float(cfg.get("signal_delay_seconds", 10.0), 10.0)
        history_limit = _as_int(cfg.get("history_limit", 1500), 1500)
        max_history = cfg.get("max_history")
        if max_history is not None:
            max_history = _as_int(max_history, 0)
        maker_offset_bps = _as_float(cfg.get("maker_offset_bps", 1.0), 1.0)
        auto_trade = bool(cfg.get("auto_trade", False))
        sync_exchange_positions = bool(cfg.get("sync_exchange_positions", group == "live"))
        sync_interval_seconds = _as_float(cfg.get("sync_interval_seconds", 10.0), 10.0)
        sync_history_path = cfg.get("sync_history_path")
        live_order_retry_seconds = _as_float(cfg.get("live_order_retry_seconds", 30.0), 30.0)
        live_order_max_attempts = _as_int(cfg.get("live_order_max_attempts", 0), 0)
        if sync_history_path is None and group == "live":
            sync_history_path = "backtest/live_position_sync.jsonl"

        api_key = str(cfg.get("api_key", "") or "")
        api_secret = str(cfg.get("api_secret", "") or "")
        if group == "live" and (not api_key or not api_secret):
            logger.warning("Live runner missing api_key/api_secret; skipping live start.")
            return None
        binance = BinanceFuturesClient(api_key, api_secret)
        dual_side_position: Optional[bool] = None
        if group == "live" and auto_trade:
            def _is_noop_position_mode(exc: Exception) -> bool:
                code, msg, body = self._extract_binance_error(exc)
                if code in {-4059}:
                    return True
                text = " ".join(
                    part for part in [msg or "", body or "", str(exc)] if part
                ).lower()
                return "no need to change position side" in text

            def _is_noop_margin_type(exc: Exception) -> bool:
                code, msg, body = self._extract_binance_error(exc)
                if code in {-4046}:
                    return True
                text = " ".join(
                    part for part in [msg or "", body or "", str(exc)] if part
                ).lower()
                return "no need to change margin type" in text or "already" in text

            try:
                binance.set_position_mode(True)
                dual_side_position = True
            except Exception as exc:
                if not _is_noop_position_mode(exc):
                    logger.warning(
                        "Live hedge mode setup failed: %s",
                        self._format_binance_error(exc),
                    )
                else:
                    dual_side_position = True
            try:
                binance.set_margin_type(symbol, "ISOLATED")
            except Exception as exc:
                if not _is_noop_margin_type(exc):
                    logger.warning(
                        "Live margin type setup failed: %s",
                        self._format_binance_error(exc),
                    )
            if leverage:
                try:
                    binance.set_leverage(symbol, int(leverage))
                except Exception:
                    pass
            try:
                payload = binance.get_position_mode()
                if isinstance(payload, dict):
                    dual_side_position = bool(payload.get("dualSidePosition"))
            except Exception:
                pass

        strategy_name = "cheatkey"
        strategy_params = {}
        if isinstance(self.settings.strategy, dict):
            raw_params = self.settings.strategy.get("params", {})
            if isinstance(raw_params, dict):
                strategy_params = dict(raw_params)

        def on_trade(group_name: str, trade: Dict[str, Any], data: pd.DataFrame) -> None:
            asyncio.create_task(self._handle_realtime_trade(group_name, trade, data))

        def on_payload(group_name: str, payload: RunnerPayload) -> None:
            self._realtime_payloads[group_name] = payload

        return RealtimeRunner(
            group=group,
            symbol=symbol,
            interval=interval,
            initial_balance=initial_balance,
            fee_rate=fee_rate,
            leverage=leverage,
            position_size=position_size,
            risk_per_trade=risk_per_trade,
            max_position_fraction_per_side=max_pos_fraction,
            slippage=slippage,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            poll_interval=poll_interval,
            signal_delay_seconds=signal_delay_seconds,
            history_limit=history_limit,
            max_history=max_history,
            binance=binance,
            maker_offset_bps=maker_offset_bps,
            auto_trade=auto_trade,
            sync_exchange_positions=sync_exchange_positions,
            sync_interval_seconds=sync_interval_seconds,
            sync_history_path=sync_history_path,
            live_order_retry_seconds=live_order_retry_seconds,
            live_order_max_attempts=live_order_max_attempts,
            dual_side_position=dual_side_position,
            on_trade=on_trade,
            on_payload=on_payload,
        )

    async def _handle_realtime_trade(
        self,
        group: str,
        trade: Dict[str, Any],
        data: pd.DataFrame,
    ) -> None:
        trade_type = str(trade.get("type", "")).lower()
        if trade_type not in {"entry", "add", "exit", "exit_partial"}:
            return
        await self._send_trade_log(group, trade)
        if trade_type == "exit":
            await self._send_trade_result(group, trade, data)

    async def _send_trade_log(self, group: str, trade: Dict[str, Any]) -> None:
        channel = await self._get_or_create_group_channel(
            group, self._group_channel_name(group, "tradelog")
        )
        if channel is None:
            return
        direction = int(trade.get("direction", 0) or 0)
        side = "롱" if direction > 0 else "숏"
        emoji = "🟢" if direction > 0 else "🔴"
        color = discord.Color.green() if direction > 0 else discord.Color.red()
        trade_type = str(trade.get("type", "")).lower()
        action = {
            "entry": "진입",
            "add": "추가매수",
            "exit": "청산",
            "exit_partial": "부분청산",
        }.get(trade_type, trade_type)
        price = float(trade.get("price", 0.0) or 0.0)
        qty = float(trade.get("qty", 0.0) or 0.0)
        leverage = float(trade.get("leverage", 1.0) or 1.0)
        notional = abs(price * qty)
        tp_price = trade.get("rr_tp_price", trade.get("tp_price"))
        sl_price = trade.get("rr_stop_price", trade.get("stop_price"))
        pnl = trade.get("pnl")
        pnl_text = f"{float(pnl):+.2f}" if pnl is not None else "n/a"
        exit_reason = trade.get("exit_reason")
        reason_text = f" | 사유: {exit_reason}" if exit_reason else ""
        buy_time = self._format_kst_time(trade.get("entry_time") or trade.get("timestamp"))
        sell_time = self._format_kst_time(trade.get("exit_time") or trade.get("timestamp"))
        if trade_type in {"entry", "add"}:
            sell_time = "n/a"
        if trade_type in {"exit", "exit_partial"}:
            buy_time = self._format_kst_time(trade.get("entry_time"))
        symbol = "UNKNOWN"
        runner = self._runner_for_group(group)
        if runner is not None:
            symbol = runner.symbol

        embed = discord.Embed(
            title=f"{emoji} {symbol} {side} {action}",
            color=color,
        )
        embed.add_field(
            name="가격/수량",
            value=f"가격: {self._fmt_price(price)}\n수량: {qty:.6g}\n금액: {notional:,.2f}",
            inline=True,
        )
        embed.add_field(
            name="레버리지/TP/SL",
            value=(
                f"{leverage:.2f}x\n"
                f"TP: {self._fmt_price(tp_price)}\n"
                f"SL: {self._fmt_price(sl_price)}"
            ),
            inline=True,
        )
        embed.add_field(
            name="시간(KST)",
            value=f"매수: {buy_time}\n매도: {sell_time}",
            inline=False,
        )
        if trade_type in {"exit", "exit_partial"}:
            ret_pct = float(trade.get("return", 0.0)) * 100.0
            embed.add_field(
                name="수익",
                value=f"수익률: {ret_pct:+.2f}%\n수익금: {pnl_text}{reason_text}",
                inline=False,
            )
        await channel.send(embed=embed)

    async def _send_trade_result(
        self,
        group: str,
        trade: Dict[str, Any],
        data: pd.DataFrame,
    ) -> None:
        channel = await self._get_or_create_group_channel(
            group, self._group_channel_name(group, "result")
        )
        if channel is None:
            return
        direction = int(trade.get("direction", 0) or 0)
        side = "롱" if direction > 0 else "숏"
        emoji = "🟢" if direction > 0 else "🔴"
        entry_price = float(trade.get("entry_price", 0.0) or 0.0)
        exit_price = float(trade.get("price", 0.0) or 0.0)
        qty = float(trade.get("qty", 0.0) or 0.0)
        leverage = float(trade.get("leverage", 1.0) or 1.0)
        notional = abs(entry_price * qty)
        ret_pct = float(trade.get("return", 0.0)) * 100.0
        min_ret = float(trade.get("min_return", 0.0)) * 100.0
        max_ret = float(trade.get("max_return", 0.0)) * 100.0
        pnl = float(trade.get("pnl", 0.0))
        exit_reason = trade.get("exit_reason", "unknown")
        entry_reason = trade.get("entry_reason", "unknown")
        tp_price = trade.get("rr_tp_price", trade.get("tp_price"))
        sl_price = trade.get("rr_stop_price", trade.get("stop_price"))
        entry_time = self._format_kst_time(trade.get("entry_time"))
        exit_time = self._format_kst_time(trade.get("exit_time") or trade.get("timestamp"))
        symbol = "UNKNOWN"
        runner = self._runner_for_group(group)
        if runner is not None:
            symbol = runner.symbol

        title = f"{side} trade | Return {ret_pct:+.2f}%"
        with tempfile.NamedTemporaryFile(
            prefix=f"{group}_trade_snapshot_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            snapshot_path = create_trade_snapshot_chart(
                data,
                trade,
                Path(tmp_file.name),
                title=title,
            )
        with tempfile.NamedTemporaryFile(
            prefix=f"{group}_trade_pnl_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            pnl_path = create_trade_pnl_curve(
                data,
                trade,
                Path(tmp_file.name),
            )
        chart_paths = [path for path in (snapshot_path, pnl_path) if path]

        result_color = discord.Color.green() if ret_pct >= 0 else discord.Color.red()
        embed = discord.Embed(
            title=f"{emoji} {symbol} {side} 결과",
            description=f"수익률 {ret_pct:+.2f}%",
            color=result_color,
        )
        embed.add_field(
            name="진입/청산",
            value=(
                f"진입: {self._fmt_price(entry_price)}\n"
                f"청산: {self._fmt_price(exit_price)}"
            ),
            inline=True,
        )
        embed.add_field(
            name="레버리지/TP/SL",
            value=(
                f"{leverage:.2f}x\n"
                f"TP: {self._fmt_price(tp_price)}\n"
                f"SL: {self._fmt_price(sl_price)}"
            ),
            inline=True,
        )
        embed.add_field(
            name="성과",
            value=(
                f"수익률: {ret_pct:+.2f}%\n"
                f"최고: {max_ret:+.2f}%\n"
                f"최저: {min_ret:+.2f}%"
            ),
            inline=True,
        )
        embed.add_field(
            name="손익",
            value=f"매수 USDT: {notional:,.2f}\n순수익: {pnl:+,.2f}",
            inline=True,
        )
        embed.add_field(
            name="시간(KST)",
            value=f"매수: {entry_time}\n매도: {exit_time}",
            inline=False,
        )
        embed.add_field(
            name="사유",
            value=f"진입: {entry_reason}\n청산: {exit_reason}",
            inline=False,
        )
        await self._send_embed_and_charts(channel.send, embed, chart_paths)

    async def _ensure_channels(self) -> None:
        guild = self._get_guild()
        if guild is None:
            return
        main_category = await self._get_or_create_category(guild, self.main_category_name)
        if self.chat_channel_name:
            await self._get_or_create_channel(guild, self.chat_channel_name, category=main_category)
        backtest_category = await self._get_or_create_category(guild, self.backtest_category_name)
        sim_category = await self._get_or_create_category(guild, self.sim_category_name)
        live_category = await self._get_or_create_category(guild, self.live_category_name)

        for name in self.backtest_channel_names:
            await self._get_or_create_channel(guild, name, category=backtest_category)
        for name in self.sim_channel_names:
            await self._get_or_create_channel(guild, name, category=sim_category)
        for name in self.live_channel_names:
            await self._get_or_create_channel(guild, name, category=live_category)

    def _get_guild(self) -> Optional[discord.Guild]:
        if self.guild_id:
            return self.get_guild(int(self.guild_id))
        if self.guilds:
            return self.guilds[0]
        return None

    async def _get_or_create_channel(
        self,
        guild: discord.Guild,
        name: str,
        category: Optional[discord.CategoryChannel] = None,
    ) -> Optional[discord.TextChannel]:
        channel = discord.utils.get(guild.text_channels, name=name)
        if channel:
            if category and channel.category_id != category.id:
                try:
                    await channel.edit(category=category)
                except discord.Forbidden:
                    pass
            return channel
        if not self.create_channels:
            return channel
        try:
            return await guild.create_text_channel(name=name, category=category)
        except discord.Forbidden:
            return channel

    async def _get_or_create_category(
        self, guild: discord.Guild, name: Optional[str]
    ) -> Optional[discord.CategoryChannel]:
        if not name:
            return None
        category = discord.utils.get(guild.categories, name=name)
        if category:
            return category
        if not self.create_channels:
            return None
        try:
            return await guild.create_category(name=name)
        except discord.Forbidden:
            return None

    def _is_command_channel(self, channel: discord.abc.GuildChannel) -> bool:
        if not self.command_channel_names:
            return True
        return getattr(channel, "name", "") in self.command_channel_names

    def _is_chat_channel(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        if channel is None:
            return False
        return getattr(channel, "name", "") == self.chat_channel_name

    def _is_sim_channel(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        if channel is None:
            return False
        name = getattr(channel, "name", "")
        return bool(name) and name in self.sim_channel_names

    def _is_live_channel(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        if channel is None:
            return False
        name = getattr(channel, "name", "")
        return bool(name) and name in self.live_channel_names

    def _is_backtest_channel(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        if channel is None:
            return False
        return self._is_command_channel(channel)

    def _backtest_channel_notice(self) -> str:
        return self._command_channel_notice("for backtests")

    def _chat_channel_notice(self) -> str:
        if self.chat_channel_name:
            return f"Please use #{self.chat_channel_name} for wordle."
        return "Please use the chat channel for wordle."

    def _command_channel_notice(self, purpose: str = "for commands") -> str:
        names = [f"#{name}" for name in self.command_channel_names if name]
        if names:
            return f"Please use {', '.join(names)} {purpose}."
        return "Please use a configured command channel for this command."

    def _forced_strategy_name(self, channel: Optional[discord.abc.GuildChannel]) -> Optional[str]:
        if self._is_sim_channel(channel) or self._is_live_channel(channel):
            return "cheatkey"
        return None

    def _enforce_cheatkey_strategy(
        self,
        channel: Optional[discord.abc.GuildChannel],
        strategy_overrides: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        forced = self._forced_strategy_name(channel)
        if not forced:
            return strategy_overrides, None
        overrides = dict(strategy_overrides)
        requested = overrides.get("name")
        if requested is not None:
            requested_name = str(requested).strip().lower().replace(" ", "_").replace("-", "_")
            if requested_name != forced:
                return (
                    overrides,
                    f"Only `{forced}` strategy is allowed in this channel.",
                )
        overrides["name"] = forced
        return overrides, None

    def _apply_forced_strategy(
        self, settings: Settings, channel: Optional[discord.abc.GuildChannel]
    ) -> Optional[str]:
        forced = self._forced_strategy_name(channel)
        if not forced:
            return None
        current = settings.strategy.get("name")
        if current is not None:
            current_name = str(current).strip().lower().replace(" ", "_").replace("-", "_")
            if current_name != forced:
                return f"Only `{forced}` strategy is allowed in this channel."
        settings.strategy["name"] = forced
        return None

    def _is_guild_owner(self, interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            return False
        return interaction.user.id == interaction.guild.owner_id

    def _is_guild_owner_ctx(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            return False
        return ctx.author.id == ctx.guild.owner_id

    def _resolve_channel_group(
        self, channel: Optional[discord.abc.GuildChannel]
    ) -> str:
        if self._is_sim_channel(channel):
            return "sim"
        if self._is_live_channel(channel):
            return "live"
        return "backtest"

    def _group_category_name(self, group: str) -> Optional[str]:
        if group == "sim":
            return self.sim_category_name
        if group == "live":
            return self.live_category_name
        return self.backtest_category_name

    def _group_channel_name(self, group: str, kind: str) -> Optional[str]:
        if kind == "result":
            if group == "sim":
                return self.sim_result_channel_name
            if group == "live":
                return self.live_result_channel_name
            return self.result_channel_name
        if kind == "status":
            if group == "sim":
                return self.sim_status_channel_name
            if group == "live":
                return self.live_status_channel_name
            return None
        if kind == "tradelog":
            if group == "sim":
                return self.sim_tradelog_channel_name
            if group == "live":
                return self.live_tradelog_channel_name
            return None
        return None

    async def _get_or_create_group_channel(
        self, group: str, name: Optional[str]
    ) -> Optional[discord.TextChannel]:
        if not name:
            return None
        guild = self._get_guild()
        if guild is None:
            return None
        category_name = self._group_category_name(group)
        category = await self._get_or_create_category(guild, category_name)
        return await self._get_or_create_channel(guild, name, category=category)

    async def _get_result_channel(
        self, channel: Optional[discord.abc.GuildChannel] = None
    ) -> Optional[discord.TextChannel]:
        group = self._resolve_channel_group(channel)
        name = self._group_channel_name(group, "result")
        if not name:
            return None
        return await self._get_or_create_group_channel(group, name)

    async def _get_status_channel(
        self, channel: Optional[discord.abc.GuildChannel] = None
    ) -> Optional[discord.TextChannel]:
        group = self._resolve_channel_group(channel)
        name = self._group_channel_name(group, "status")
        if not name:
            return None
        return await self._get_or_create_group_channel(group, name)

    async def _get_tradelog_channel(
        self, channel: Optional[discord.abc.GuildChannel] = None
    ) -> Optional[discord.TextChannel]:
        group = self._resolve_channel_group(channel)
        name = self._group_channel_name(group, "tradelog")
        if not name:
            return None
        return await self._get_or_create_group_channel(group, name)

    async def _get_data_list_channel(self) -> Optional[discord.TextChannel]:
        guild = self._get_guild()
        if guild is None or not self.data_list_channel_name:
            return None
        channel = discord.utils.get(guild.text_channels, name=self.data_list_channel_name)
        if channel:
            return channel
        if not self.create_channels:
            return None
        return await self._get_or_create_channel(guild, self.data_list_channel_name)

    async def _get_strategy_list_channel(self) -> Optional[discord.TextChannel]:
        guild = self._get_guild()
        if guild is None or not self.strategy_list_channel_name:
            return None
        channel = discord.utils.get(guild.text_channels, name=self.strategy_list_channel_name)
        if channel:
            return channel
        if not self.create_channels:
            return None
        return await self._get_or_create_channel(guild, self.strategy_list_channel_name)

    def _price_data_folder(self) -> Path:
        folder = self.settings.data.get("folder", "price data")
        return Path(folder)

    def _list_price_files(self) -> List[str]:
        folder = self._price_data_folder()
        if not folder.exists():
            return []
        return sorted([p.name for p in folder.iterdir() if p.is_file()])

    def _resolve_price_file(self, filename: str) -> Optional[Path]:
        if not filename:
            return None
        files = self._list_price_files()
        if filename not in files:
            return None
        return self._price_data_folder() / filename

    def _get_data_list_message_ids(self) -> Dict[str, int]:
        stored = self.settings.discord.get("data_list_message_ids", {})
        if not isinstance(stored, dict):
            return {}
        message_ids: Dict[str, int] = {}
        for key, value in stored.items():
            try:
                message_ids[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return message_ids

    def _get_strategy_list_message_ids(self) -> Dict[str, int]:
        stored = self.settings.discord.get("strategy_list_message_ids", {})
        if not isinstance(stored, dict):
            return {}
        message_ids: Dict[str, int] = {}
        for key, value in stored.items():
            try:
                message_ids[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return message_ids

    def _set_data_list_message_id(self, guild_id: int, message_id: int) -> None:
        if "data_list_message_ids" not in self.settings.discord:
            self.settings.discord["data_list_message_ids"] = {}
        if not isinstance(self.settings.discord["data_list_message_ids"], dict):
            self.settings.discord["data_list_message_ids"] = {}
        self.settings.discord["data_list_message_ids"][str(guild_id)] = int(message_id)
        save_settings(self.settings_path, self.settings)

    def _set_strategy_list_message_id(self, guild_id: int, message_id: int) -> None:
        if "strategy_list_message_ids" not in self.settings.discord:
            self.settings.discord["strategy_list_message_ids"] = {}
        if not isinstance(self.settings.discord["strategy_list_message_ids"], dict):
            self.settings.discord["strategy_list_message_ids"] = {}
        self.settings.discord["strategy_list_message_ids"][str(guild_id)] = int(message_id)
        save_settings(self.settings_path, self.settings)

    def _build_data_list_message(self) -> str:
        files = self._list_price_files()
        if not files:
            return "No data files yet."

        grouped: Dict[str, List[str]] = {}
        for name in files:
            stem = name.rsplit(".", 1)[0]
            parts = stem.split("_")
            if len(parts) < 4:
                grouped.setdefault("UNKNOWN", []).append(name)
                continue
            symbol = "_".join(parts[:-3])
            interval = parts[-3]
            start = parts[-2]
            end = parts[-1]
            entry = f"{interval} {start}-{end}"
            grouped.setdefault(symbol, []).append(entry)

        lines: list[str] = [f"Data files ({len(files)} total):"]
        for symbol in sorted(grouped.keys()):
            lines.append(f"[{symbol}]")
            entries = sorted(set(grouped[symbol]))
            for entry in entries:
                lines.append(f"- {entry}")

        content = "\n".join(lines)
        if len(content) <= 1900:
            return content
        truncated = content[:1860].rsplit("\n", 1)[0]
        return f"{truncated}\n... (truncated)"

    def _format_data_label(self, filename: str) -> str:
        stem = filename.rsplit(".", 1)[0]
        parts = stem.split("_")
        if len(parts) < 4:
            return filename
        symbol = "_".join(parts[:-3])
        interval = parts[-3]
        start = parts[-2]
        end = parts[-1]
        return f"{symbol} | {interval} | {start}-{end}"

    def _build_strategy_list_message(self) -> str:
        strategies = self._list_strategy_names()
        if not strategies:
            return "No strategies found."

        lines: List[str] = [f"Strategies ({len(strategies)} total):"]
        for name in strategies:
            lines.append(f"- {name}")

        files = self._list_price_files()
        if not files:
            lines.append("")
            lines.append("No data files yet.")
            content = "\n".join(lines)
            return content[:1900]

        lines.append("")
        lines.append("Data performance (Return / MaxDD / WinRate / Trades / PF):")
        for filename in files:
            label = self._format_data_label(filename)
            lines.append(f"[{label}]")
            for name in strategies:
                try:
                    stats = run_backtest_summary(
                        self.settings,
                        data_overrides={"file_pattern": filename},
                        strategy_name=name,
                    )
                    total_return = float(stats.get("total_return", 0.0)) * 100.0
                    max_drawdown = float(stats.get("max_drawdown", 0.0)) * 100.0
                    win_rate = float(stats.get("win_rate", 0.0)) * 100.0
                    total_trades = int(stats.get("total_trades", 0))
                    profit_factor = stats.get("profit_factor", 0.0)
                    pf_text = (
                        "inf"
                        if profit_factor == float("inf")
                        else f"{float(profit_factor):.2f}"
                    )
                    lines.append(
                        f"- {name}: {total_return:+.2f}%, DD {max_drawdown:.2f}%, "
                        f"WR {win_rate:.1f}%, T {total_trades}, PF {pf_text}"
                    )
                except Exception as exc:
                    lines.append(f"- {name}: error ({type(exc).__name__})")

        content = "\n".join(lines)
        if len(content) <= 1900:
            return content
        truncated = content[:1860].rsplit("\n", 1)[0]
        return f"{truncated}\n... (truncated)"

    async def _update_data_list_channel(self) -> None:
        channel = await self._get_data_list_channel()
        if channel is None:
            return

        content = self._build_data_list_message()
        guild = channel.guild
        message_ids = self._get_data_list_message_ids()
        message_id = message_ids.get(str(guild.id))
        if message_id:
            try:
                message = await channel.fetch_message(message_id)
                await message.edit(content=content)
                return
            except (discord.NotFound, discord.Forbidden):
                message_id = None

        try:
            message = await channel.send(content)
        except discord.Forbidden:
            return
        self._set_data_list_message_id(guild.id, message.id)

    async def _update_strategy_list_channel(self) -> None:
        channel = await self._get_strategy_list_channel()
        if channel is None:
            return

        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self._build_strategy_list_message)
        except Exception as exc:
            content = f"Failed to build strategy list: {type(exc).__name__}"

        guild = channel.guild
        message_ids = self._get_strategy_list_message_ids()
        message_id = message_ids.get(str(guild.id))
        if message_id:
            try:
                message = await channel.fetch_message(message_id)
                await message.edit(content=content)
                return
            except (discord.NotFound, discord.Forbidden):
                message_id = None

        try:
            message = await channel.send(content)
        except discord.Forbidden:
            return
        self._set_strategy_list_message_id(guild.id, message.id)

    def _list_strategy_names(self) -> List[str]:
        return list_strategy_names()

    def _get_download_defaults(self) -> Dict[str, str]:
        defaults = self.settings.discord.get("download_defaults", {})
        if not isinstance(defaults, dict):
            return {}
        return {key: str(value) for key, value in defaults.items()}

    def _set_download_defaults(self, values: Dict[str, str]) -> None:
        if "download_defaults" not in self.settings.discord:
            self.settings.discord["download_defaults"] = {}
        if not isinstance(self.settings.discord["download_defaults"], dict):
            self.settings.discord["download_defaults"] = {}
        self.settings.discord["download_defaults"].update(values)
        save_settings(self.settings_path, self.settings)

    async def _data_file_autocomplete(
        self, _: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        matches = []
        for name in self._list_price_files():
            if current.lower() in name.lower():
                matches.append(app_commands.Choice(name=name, value=name))
            if len(matches) >= 25:
                break
        return matches

    async def _strategy_autocomplete(
        self, _: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        matches = []
        for name in self._list_strategy_names():
            if current.lower() in name.lower():
                matches.append(app_commands.Choice(name=name, value=name))
            if len(matches) >= 25:
                break
        return matches

    def _supported_intervals(self) -> List[str]:
        return [
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
        ]

    async def _interval_autocomplete(
        self, _: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        matches = []
        for name in self._supported_intervals():
            if current.lower() in name.lower():
                matches.append(app_commands.Choice(name=name, value=name))
            if len(matches) >= 25:
                break
        return matches

    def _coerce_scalar(self, value: str) -> Any:
        lowered = value.strip().lower()
        if lowered in {"null", "none"}:
            return None
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _coerce_value(self, value: str) -> Any:
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if not inner:
                return []
            items = [item.strip() for item in inner.split(",") if item.strip()]
            return [self._coerce_scalar(item) for item in items]
        return self._coerce_scalar(stripped)

    def _parse_kv_tokens(self, tokens: Sequence[str]) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for token in tokens:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key:
                parsed[key] = value
        return parsed

    def _coerce_backtest_arg(self, key: str, value: str) -> Any:
        float_keys = {"initial_balance", "fee_rate", "leverage", "position_size", "slippage"}
        int_keys = {"fast_window", "slow_window"}
        if key in float_keys:
            return float(value)
        if key in int_keys:
            return int(value)
        return value

    def _apply_setting(self, section: str, key: str, value: Any) -> None:
        target = getattr(self.settings, section)
        parts = key.split(".")
        current: Dict[str, Any] = target
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _get_setting_value(self, section: str, key: str) -> Any:
        target = getattr(self.settings, section)
        parts = key.split(".") if key else []
        current: Any = target
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _get_setting_value_from(self, settings: Settings, section: str, key: str) -> Any:
        target = getattr(settings, section)
        parts = key.split(".") if key else []
        current: Any = target
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _apply_setting_value_to(
        self,
        settings: Settings,
        section: str,
        key: str,
        value: Any,
        index: Optional[int] = None,
    ) -> bool:
        target = getattr(settings, section)
        base_key = key
        if index is not None:
            base_key = key
        parts = base_key.split(".")
        current: Dict[str, Any] = target
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        last = parts[-1]
        if index is None:
            current[last] = value
            return True
        existing = current.get(last)
        if not isinstance(existing, list):
            return False
        if index < 0 or index >= len(existing):
            return False
        updated = list(existing)
        updated[index] = value
        current[last] = updated
        return True

    def _get_realtime_cfg(self, group: str) -> Dict[str, Any]:
        key = "sim" if group == "sim" else "live"
        cfg = self.settings.discord.get(key)
        if not isinstance(cfg, dict):
            cfg = {}
            self.settings.discord[key] = cfg
        if group == "sim":
            self.sim_cfg = cfg
        else:
            self.live_cfg = cfg
        return cfg

    def _get_live_client(self) -> tuple[Optional[BinanceFuturesClient], Optional[str]]:
        cfg = self._get_realtime_cfg("live")
        api_key = str(cfg.get("api_key", "") or "")
        api_secret = str(cfg.get("api_secret", "") or "")
        if not api_key or not api_secret:
            return None, "Live API key/secret not configured. Use `!live api` first."
        runner = self.live_runner
        if runner and runner.binance:
            return runner.binance, None
        return BinanceFuturesClient(api_key, api_secret), None

    def _get_spot_client(self) -> tuple[Optional[BinanceSpotClient], Optional[str]]:
        cfg = self._get_realtime_cfg("live")
        api_key = str(cfg.get("api_key", "") or "")
        api_secret = str(cfg.get("api_secret", "") or "")
        if not api_key or not api_secret:
            return None, "Live API key/secret not configured. Use `!live api` first."
        return BinanceSpotClient(api_key, api_secret), None

    async def _fetch_live_account_snapshot(
        self,
        client: BinanceFuturesClient,
        symbol: str,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
        available = None
        wallet = None
        notes: List[str] = []
        try:
            account = await asyncio.to_thread(client.get_account)
            if isinstance(account, dict):
                assets = account.get("assets") or []
                for asset in assets:
                    if str(asset.get("asset", "")).upper() == "USDT":
                        try:
                            available = float(asset.get("availableBalance", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            available = None
                        try:
                            wallet = float(asset.get("walletBalance", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            wallet = None
                        break
        except Exception as exc:
            notes.append(f"Balance fetch failed: {exc}")

        price = None
        payload = self._realtime_payloads.get("live")
        if payload and payload.symbol == symbol and payload.price:
            price = float(payload.price)
        else:
            try:
                price = float(await asyncio.to_thread(client.get_price, symbol))
            except Exception as exc:
                notes.append(f"Price fetch failed: {exc}")

        note = "\n".join(notes) if notes else None
        return available, wallet, price, note

    @staticmethod
    def _apply_maker_offset(price: float, side: str, maker_offset_bps: float) -> float:
        if price <= 0:
            return price
        offset = float(maker_offset_bps or 0.0) / 10000.0
        if offset <= 0:
            return price
        if side.upper() == "BUY":
            return price * (1.0 - offset)
        return price * (1.0 + offset)

    @staticmethod
    def _price_bounds(
        reference_price: float, filters: Optional[Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        if filters is None:
            return None, None
        min_price = getattr(filters, "min_price", None)
        max_price = getattr(filters, "max_price", None)
        percent_down = getattr(filters, "percent_down", None)
        percent_up = getattr(filters, "percent_up", None)
        if reference_price > 0:
            if percent_down is not None and percent_down > 0:
                candidate = reference_price * percent_down
                min_price = candidate if min_price is None else max(min_price, candidate)
            if percent_up is not None and percent_up > 0:
                candidate = reference_price * percent_up
                max_price = candidate if max_price is None else min(max_price, candidate)
        return min_price, max_price

    @staticmethod
    def _extract_binance_error(
        exc: Exception,
    ) -> tuple[Optional[int], Optional[str], Optional[str]]:
        code = None
        msg = None
        body = None
        if isinstance(exc, BinanceAPIError):
            return exc.code, exc.msg, exc.response_text
        resp = getattr(exc, "response", None)
        if resp is None:
            return code, msg, body
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

    @classmethod
    def _format_binance_error(cls, exc: Exception) -> str:
        code, msg, body = cls._extract_binance_error(exc)
        parts: List[str] = []
        if code is not None:
            parts.append(f"code={code}")
        if msg:
            parts.append(f"msg={msg}")
        if body and (not msg or body.strip() != msg.strip()):
            parts.append(f"body={body}")
        if parts:
            return ", ".join(parts)
        return str(exc)

    @staticmethod
    def _interaction_is_expired(interaction: discord.Interaction) -> bool:
        check = getattr(interaction, "is_expired", None)
        if callable(check):
            try:
                return bool(check())
            except Exception:
                return False
        expired = getattr(interaction, "expired", None)
        return bool(expired) if isinstance(expired, bool) else False

    @classmethod
    def _is_post_only_reject(cls, exc: Exception) -> bool:
        code, msg, body = cls._extract_binance_error(exc)
        if code in {-5022, -2010}:
            return True
        text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
        return "post only" in text or "post-only" in text or "immediately trigger" in text

    @classmethod
    def _is_reduce_only_not_required(cls, exc: Exception) -> bool:
        code, msg, body = cls._extract_binance_error(exc)
        if code != -1106:
            return False
        text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
        return "reduceonly" in text and "not required" in text

    @classmethod
    def _is_filter_error(cls, exc: Exception) -> bool:
        code, msg, body = cls._extract_binance_error(exc)
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

    @classmethod
    def _extract_limit_price_bound(
        cls, exc: Exception
    ) -> Tuple[Optional[float], Optional[float]]:
        _, msg, body = cls._extract_binance_error(exc)
        text = " ".join(part for part in [msg or "", body or "", str(exc)] if part).lower()
        upper_match = re.search(r"limit price can't be higher than\s*([0-9]*\.?[0-9]+)", text)
        lower_match = re.search(r"limit price can't be lower than\s*([0-9]*\.?[0-9]+)", text)
        upper = float(upper_match.group(1)) if upper_match else None
        lower = float(lower_match.group(1)) if lower_match else None
        return lower, upper

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

    async def _ensure_hedge_and_isolated(
        self,
        client: BinanceFuturesClient,
        symbol: str,
    ) -> Tuple[List[str], bool, bool]:
        def _is_noop_position_mode(exc: Exception) -> bool:
            code, msg, body = self._extract_binance_error(exc)
            if code in {-4059}:
                return True
            text = " ".join(
                part for part in [msg or "", body or "", str(exc)] if part
            ).lower()
            if "no need to change position side" in text:
                return True
            if "position side" in text and "no need to change" in text:
                return True
            return False

        def _is_noop_margin_type(exc: Exception) -> bool:
            code, msg, body = self._extract_binance_error(exc)
            if code in {-4046}:
                return True
            text = " ".join(
                part for part in [msg or "", body or "", str(exc)] if part
            ).lower()
            if "no need to change margin type" in text:
                return True
            if "margin type" in text and "already" in text:
                return True
            if "no need to change margin type" in text or "already" in text:
                return True
            return False

        notes: List[str] = []
        hedge_ok = False
        try:
            await asyncio.to_thread(client.set_position_mode, True)
            hedge_ok = True
        except Exception as exc:
            if _is_noop_position_mode(exc):
                hedge_ok = True
            else:
                try:
                    payload = await asyncio.to_thread(client.get_position_mode)
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("dualSidePosition") is True:
                    hedge_ok = True
                else:
                    notes.append(
                        f"Failed to set hedge mode: {self._format_binance_error(exc)}"
                    )
        isolated_ok = False
        try:
            await asyncio.to_thread(client.set_margin_type, symbol, "ISOLATED")
            isolated_ok = True
        except Exception as exc:
            if _is_noop_margin_type(exc):
                isolated_ok = True
            else:
                try:
                    payload = await asyncio.to_thread(client.get_position_risk, symbol)
                except Exception:
                    payload = None
                margin_type = None
                if isinstance(payload, list) and payload:
                    margin_type = payload[0].get("marginType")
                elif isinstance(payload, dict):
                    margin_type = payload.get("marginType")
                if isinstance(margin_type, str) and margin_type.upper() == "ISOLATED":
                    isolated_ok = True
                else:
                    notes.append(
                        f"Failed to set isolated margin: {self._format_binance_error(exc)}"
                    )
        return notes, hedge_ok, isolated_ok

    async def _place_limit_maker_with_retry(
        self,
        client: BinanceFuturesClient,
        symbol: str,
        side: str,
        qty: float,
        base_price: float,
        reduce_only: bool,
        position_side: str,
        maker_offset_bps: float,
        retry_seconds: float,
        max_attempts: int,
    ) -> Dict[str, Any]:
        tick_size = 0.0
        step_size = 0.0
        filters: Optional[Any] = None
        try:
            filters = await asyncio.to_thread(client.get_symbol_filters, symbol)
            tick_size = filters.tick_size
            step_size = filters.step_size
        except Exception:
            pass

        remaining_qty = RealtimeRunner._round_step(qty, step_size)
        if remaining_qty <= 0:
            return {
                "filled": False,
                "remaining_qty": remaining_qty,
                "order_id": None,
                "note": "Quantity too small after rounding.",
            }

        attempt = 0
        last_order_id: Optional[int] = None
        post_only_rejects = 0
        filters_refreshed = False
        last_error_note: Optional[str] = None
        forced_min_price: Optional[float] = None
        forced_max_price: Optional[float] = None
        reduce_only_param = reduce_only
        while remaining_qty > 0:
            attempt += 1
            if max_attempts > 0 and attempt > max_attempts:
                break

            target_price = base_price
            reference_price = base_price
            if attempt > 1:
                try:
                    target_price = float(
                        await asyncio.to_thread(client.get_price, symbol)
                    )
                    reference_price = target_price
                except Exception:
                    target_price = base_price
                    reference_price = target_price
            else:
                try:
                    reference_price = float(
                        await asyncio.to_thread(client.get_price, symbol)
                    )
                except Exception:
                    reference_price = target_price
            adjusted_price = self._apply_maker_offset(
                float(target_price), side, maker_offset_bps
            )
            if post_only_rejects > 0:
                nudge = tick_size if tick_size > 0 else adjusted_price * 0.0001
                if nudge > 0:
                    adjusted_price = (
                        adjusted_price - (nudge * post_only_rejects)
                        if side.upper() == "BUY"
                        else adjusted_price + (nudge * post_only_rejects)
                    )
            min_price, max_price = self._merge_price_bounds(
                self._price_bounds(float(reference_price), filters),
                forced_min_price,
                forced_max_price,
            )
            adjusted_price = RealtimeRunner._round_price_with_bounds(
                adjusted_price, tick_size, side, min_price, max_price
            )
            if adjusted_price <= 0:
                return {
                    "filled": False,
                    "remaining_qty": remaining_qty,
                    "order_id": last_order_id,
                    "note": "Price invalid after rounding.",
                }

            try:
                qty_param = RealtimeRunner._format_by_step(remaining_qty, step_size)
                price_param = RealtimeRunner._format_by_step(adjusted_price, tick_size)
                order = await asyncio.to_thread(
                    client.place_limit_maker,
                    symbol,
                    side,
                    qty_param,
                    price_param,
                    reduce_only_param,
                    position_side,
                )
            except Exception as exc:
                bound_min, bound_max = self._extract_limit_price_bound(exc)
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
                    last_error_note = f"Order failed: {self._format_binance_error(exc)}"
                    continue
                if self._is_reduce_only_not_required(exc) and reduce_only_param:
                    reduce_only_param = False
                    last_error_note = f"Order failed: {self._format_binance_error(exc)}"
                    continue
                if self._is_filter_error(exc) and not filters_refreshed:
                    filters_refreshed = True
                    try:
                        filters = await asyncio.to_thread(
                            client.get_symbol_filters, symbol
                        )
                        tick_size = filters.tick_size
                        step_size = filters.step_size
                        remaining_qty = RealtimeRunner._round_step(
                            remaining_qty, step_size
                        )
                        if remaining_qty <= 0:
                            return {
                                "filled": False,
                                "remaining_qty": remaining_qty,
                                "order_id": last_order_id,
                                "note": "Quantity too small after rounding.",
                            }
                    except Exception:
                        pass
                    last_error_note = f"Order failed: {self._format_binance_error(exc)}"
                    continue
                if self._is_post_only_reject(exc):
                    post_only_rejects += 1
                    last_error_note = f"Order failed: {self._format_binance_error(exc)}"
                    continue
                return {
                    "filled": False,
                    "remaining_qty": remaining_qty,
                    "order_id": last_order_id,
                    "note": f"Order failed: {self._format_binance_error(exc)}",
                }

            order_id = order.get("orderId") if isinstance(order, dict) else None
            last_order_id = order_id
            if retry_seconds <= 0 or order_id is None:
                return {
                    "filled": True,
                    "remaining_qty": remaining_qty,
                    "order_id": order_id,
                    "note": "Order placed (status unknown; retry disabled).",
                }

            await asyncio.sleep(retry_seconds)

            status = ""
            executed_qty = 0.0
            try:
                order_status = await asyncio.to_thread(
                    client.get_order, symbol, order_id
                )
                if isinstance(order_status, dict):
                    status = str(order_status.get("status", "")).upper()
                    executed_qty = float(order_status.get("executedQty", 0.0) or 0.0)
            except Exception as exc:
                return {
                    "filled": False,
                    "remaining_qty": remaining_qty,
                    "order_id": order_id,
                    "note": f"Order status failed: {self._format_binance_error(exc)}",
                }

            if status == "FILLED":
                return {
                    "filled": True,
                    "remaining_qty": 0.0,
                    "order_id": order_id,
                    "note": None,
                }

            try:
                await asyncio.to_thread(client.cancel_order, symbol, order_id)
            except Exception:
                pass

            remaining_qty = max(remaining_qty - executed_qty, 0.0)
            remaining_qty = RealtimeRunner._round_step(remaining_qty, step_size)

        return {
            "filled": False,
            "remaining_qty": remaining_qty,
            "order_id": last_order_id,
            "note": last_error_note or "Order not fully filled after retries.",
        }

    async def _execute_live_order(
        self,
        target: Union[discord.Interaction, commands.Context],
        draft: LiveOrderDraft,
    ) -> None:
        async def _send(message: str, *, ephemeral: bool = False) -> None:
            if isinstance(target, discord.Interaction):
                channel = getattr(target, "channel", None)
                if self._interaction_is_expired(target):
                    if channel is not None:
                        await channel.send(message)
                    return
                try:
                    if target.response.is_done():
                        await target.followup.send(message, ephemeral=ephemeral)
                    else:
                        await target.response.send_message(message, ephemeral=ephemeral)
                    return
                except (discord.NotFound, discord.errors.NotFound, discord.HTTPException) as exc:
                    if getattr(exc, "code", None) not in {None, 10062}:
                        raise
                    try:
                        await target.followup.send(message, ephemeral=ephemeral)
                        return
                    except Exception:
                        if channel is not None:
                            await channel.send(message)
            else:
                await target.reply(message)

        client, err = self._get_live_client()
        if err:
            await _send(err, ephemeral=True)
            return

        cfg = self._get_realtime_cfg("live")
        retry_seconds = float(cfg.get("live_order_retry_seconds", 30.0) or 0.0)
        max_attempts = int(cfg.get("live_order_max_attempts", 0) or 0)

        notes, hedge_ok, _ = await self._ensure_hedge_and_isolated(client, draft.symbol)
        if not hedge_ok:
            extra = f"\nNote: {' | '.join(notes)}" if notes else ""
            await _send(
                "Live order blocked: hedge mode is not enabled on the account."
                f"{extra}",
                ephemeral=True,
            )
            return

        leverage_note = ""
        try:
            await asyncio.to_thread(client.set_leverage, draft.symbol, int(draft.leverage))
        except Exception as exc:
            leverage_note = f"Leverage set failed: {self._format_binance_error(exc)}"

        qty = draft.qty
        if draft.usdt_amount > 0 and draft.price > 0 and draft.leverage > 0:
            expected_qty = (draft.usdt_amount * draft.leverage) / draft.price
            if expected_qty > 0:
                diff = abs(qty - expected_qty) / expected_qty if qty > 0 else 1.0
                if qty <= 0 or diff > 0.01:
                    qty = expected_qty
        min_notional = None
        try:
            filters = await asyncio.to_thread(client.get_symbol_filters, draft.symbol)
            min_notional = filters.min_notional
        except Exception:
            pass
        if min_notional and draft.price > 0:
            rounded_notional = qty * draft.price
            if rounded_notional < min_notional:
                await _send(
                    f"Live order blocked: notional must be at least {min_notional:,.4g} "
                    f"USDT (current {rounded_notional:,.4g} USDT).",
                    ephemeral=True,
                )
                return

        result = await self._place_limit_maker_with_retry(
            client=client,
            symbol=draft.symbol,
            side=draft.side,
            qty=qty,
            base_price=draft.price,
            reduce_only=draft.reduce_only,
            position_side=draft.position_side,
            maker_offset_bps=draft.maker_offset_bps,
            retry_seconds=retry_seconds,
            max_attempts=max_attempts,
        )

        status = "filled" if result.get("filled") else "pending/partial"
        order_id = result.get("order_id")
        note = result.get("note")
        extra_notes = [n for n in notes if n]
        if leverage_note:
            extra_notes.append(leverage_note)
        if note:
            extra_notes.append(note)
        extras = f"\nNote: {' | '.join(extra_notes)}" if extra_notes else ""

        suffix = f" order_id={order_id}" if order_id else ""
        remaining_qty = float(result.get("remaining_qty", 0.0) or 0.0)
        remaining_text = (
            f"\nRemaining qty: {remaining_qty:.6g}" if remaining_qty > 0 else ""
        )
        await _send(
            f"Live order {status}: {draft.position_side} {qty:.6g} {draft.symbol} "
            f"@ {self._fmt_price(draft.price)} (lev {draft.leverage:.2f}x){suffix}"
            f"{remaining_text}{extras}",
            ephemeral=isinstance(target, discord.Interaction),
        )

    async def _execute_close_position(
        self,
        target: Union[discord.Interaction, commands.Context],
        draft: ClosePositionDraft,
    ) -> None:
        async def _send(message: str, *, ephemeral: bool = False) -> None:
            if isinstance(target, discord.Interaction):
                channel = getattr(target, "channel", None)
                if self._interaction_is_expired(target):
                    if channel is not None:
                        await channel.send(message)
                    return
                try:
                    if target.response.is_done():
                        await target.followup.send(message, ephemeral=ephemeral)
                    else:
                        await target.response.send_message(message, ephemeral=ephemeral)
                    return
                except (discord.NotFound, discord.errors.NotFound, discord.HTTPException) as exc:
                    if getattr(exc, "code", None) not in {None, 10062}:
                        raise
                    try:
                        await target.followup.send(message, ephemeral=ephemeral)
                        return
                    except Exception:
                        if channel is not None:
                            await channel.send(message)
            else:
                await target.reply(message)

        client, err = self._get_live_client()
        if err:
            await _send(err, ephemeral=True)
            return

        cfg = self._get_realtime_cfg("live")
        retry_seconds = float(cfg.get("live_order_retry_seconds", 30.0) or 0.0)
        max_attempts = int(cfg.get("live_order_max_attempts", 0) or 0)

        notes, hedge_ok, _ = await self._ensure_hedge_and_isolated(client, draft.symbol)
        if not hedge_ok:
            extra = f"\nNote: {' | '.join(notes)}" if notes else ""
            await _send(
                "Close blocked: hedge mode is not enabled on the account."
                f"{extra}",
                ephemeral=True,
            )
            return

        result = await self._place_limit_maker_with_retry(
            client=client,
            symbol=draft.symbol,
            side=draft.side,
            qty=draft.qty,
            base_price=draft.price,
            reduce_only=True,
            position_side=draft.position_side,
            maker_offset_bps=draft.maker_offset_bps,
            retry_seconds=retry_seconds,
            max_attempts=max_attempts,
        )

        status = "filled" if result.get("filled") else "pending/partial"
        order_id = result.get("order_id")
        note = result.get("note")
        extra_notes = [n for n in notes if n]
        if note:
            extra_notes.append(note)
        extras = f"\nNote: {' | '.join(extra_notes)}" if extra_notes else ""

        suffix = f" order_id={order_id}" if order_id else ""
        remaining_qty = float(result.get("remaining_qty", 0.0) or 0.0)
        remaining_text = (
            f"\nRemaining qty: {remaining_qty:.6g}" if remaining_qty > 0 else ""
        )
        await _send(
            f"Close order {status}: {draft.position_side} {draft.qty:.6g} {draft.symbol} "
            f"@ {self._fmt_price(draft.price)}{suffix}{remaining_text}{extras}",
            ephemeral=isinstance(target, discord.Interaction),
        )


    def _sim_initial_balance(self) -> float:
        cfg = self._get_realtime_cfg("sim")
        value = cfg.get("initial_balance", self.settings.backtest.get("initial_balance"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _set_realtime_enabled(self, group: str, enabled: bool) -> None:
        cfg = self._get_realtime_cfg(group)
        cfg["enabled"] = bool(enabled)
        if group == "sim":
            self.sim_enabled = bool(enabled)
        else:
            self.live_enabled = bool(enabled)
        save_settings(self.settings_path, self.settings)

    def _stop_realtime_runner(self, group: str) -> None:
        if group == "sim":
            runner = self.sim_runner
            self.sim_runner = None
        else:
            runner = self.live_runner
            self.live_runner = None
        if runner is not None:
            runner.stop()

    def _start_realtime_runner(self, group: str) -> Optional[str]:
        if group == "sim":
            if self.sim_runner is not None:
                self.sim_runner.start()
                return None
        else:
            if self.live_runner is not None:
                self.live_runner.start()
                return None
        runner = self._build_realtime_runner(group)
        if runner is None:
            return "Failed to start runner (missing config or credentials)."
        if group == "sim":
            self.sim_runner = runner
        else:
            self.live_runner = runner
        runner.start()
        return None

    @staticmethod
    def _parse_realtime_action(args: str) -> Optional[str]:
        tokens = shlex.split(args or "")
        if not tokens:
            return "status"
        action = tokens[0].lower()
        if action in {"on", "start", "enable", "enabled", "true", "1"}:
            return "on"
        if action in {"off", "stop", "disable", "disabled", "false", "0"}:
            return "off"
        if action in {"status", "state", "show"}:
            return "status"
        if action in {"restart", "reload"}:
            return "restart"
        return None

    def _parse_grid_var(
        self, settings: Settings, token: str
    ) -> tuple[Optional[dict], Optional[str]]:
        if "." not in token:
            return None, "Use section.key format (data/backtest/strategy)."
        section, key = token.split(".", 1)
        section = section.strip()
        key = key.strip()
        if section not in {"data", "backtest", "strategy"}:
            return None, "Section must be one of: data, backtest, strategy."
        index: Optional[int] = None
        base_key = key
        if "[" in key and key.endswith("]"):
            base_key, index_part = key.rsplit("[", 1)
            index_text = index_part[:-1]
            if not index_text.isdigit():
                return None, "List index must be a number like [1]."
            index = int(index_text) - 1
            current_list = self._get_setting_value_from(settings, section, base_key)
            if not isinstance(current_list, list):
                return None, "That key is not a list."
            if index < 0 or index >= len(current_list):
                return None, "List index is out of range."
        else:
            current_value = self._get_setting_value_from(settings, section, key)
            if current_value is None:
                return None, "That key does not exist in settings."
        return {
            "section": section,
            "key": base_key,
            "index": index,
            "label": token,
        }, None

    def _build_grid_values(self, start: Any, end: Any, step: Any) -> tuple[Optional[List[Any]], Optional[str]]:
        try:
            start_val = float(start)
            end_val = float(end)
            step_val = float(step)
        except ValueError:
            return None, "Start/end/step must be numbers."
        if step_val == 0:
            return None, "Step cannot be 0."
        if start_val < end_val and step_val < 0:
            return None, "Step must be positive when start < end."
        if start_val > end_val and step_val > 0:
            return None, "Step must be negative when start > end."

        values: List[float] = []
        current = start_val
        if step_val > 0:
            while current <= end_val + 1e-12:
                values.append(current)
                current += step_val
        else:
            while current >= end_val - 1e-12:
                values.append(current)
                current += step_val

        is_int_range = all(
            float(val).is_integer() for val in [start_val, end_val, step_val]
        )
        if is_int_range:
            return [int(round(val)) for val in values], None
        return values, None

    def _run_grid_backtests(
        self,
        base_settings: Settings,
        var1: dict,
        values1: List[Any],
        var2: Optional[dict],
        values2: Optional[List[Any]],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        total_runs: Optional[int] = None,
    ) -> dict:
        results: dict = {
            "rows": {},
            "best": None,
            "worst": None,
        }
        total = total_runs or (len(values1) * (len(values2) if values2 else 1))
        completed = 0
        if progress_cb:
            progress_cb(0, total)
        for value1 in values1:
            row: dict = {}
            run_values2 = values2 or [None]
            for value2 in run_values2:
                settings_copy = copy.deepcopy(base_settings)
                applied = self._apply_setting_value_to(
                    settings_copy,
                    var1["section"],
                    var1["key"],
                    value1,
                    var1["index"],
                )
                if not applied:
                    row[value2] = {"error": "Failed to apply setting"}
                    continue
                if var2 is not None and value2 is not None:
                    applied = self._apply_setting_value_to(
                        settings_copy,
                        var2["section"],
                        var2["key"],
                        value2,
                        var2["index"],
                    )
                    if not applied:
                        row[value2] = {"error": "Failed to apply setting"}
                        continue
                try:
                    stats = run_backtest_summary(settings_copy)
                    row[value2] = {"stats": stats}
                    total_return = float(stats.get("total_return", 0.0))
                    entry = {
                        "return": total_return,
                        "value1": value1,
                        "value2": value2,
                        "stats": stats,
                    }
                    best = results["best"]
                    worst = results["worst"]
                    if best is None or total_return > best["return"]:
                        results["best"] = entry
                    if worst is None or total_return < worst["return"]:
                        results["worst"] = entry
                except Exception as exc:
                    row[value2] = {"error": f"{type(exc).__name__}"}
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)
            results["rows"][value1] = row
        return results

    def _run_comp_backtests(
        self,
        base_settings: Settings,
        var1: dict,
        values1: List[Any],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        total_runs: Optional[int] = None,
    ) -> List[dict]:
        results: List[dict] = []
        total = total_runs or len(values1)
        completed = 0
        if progress_cb:
            progress_cb(0, total)
        for value1 in values1:
            settings_copy = copy.deepcopy(base_settings)
            applied = self._apply_setting_value_to(
                settings_copy,
                var1["section"],
                var1["key"],
                value1,
                var1["index"],
            )
            if not applied:
                results.append({"value": value1, "error": "Failed to apply setting"})
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)
                continue
            chart_cfg = settings_copy.backtest.get("chart")
            if isinstance(chart_cfg, dict):
                chart_cfg = dict(chart_cfg)
                chart_cfg["enabled"] = False
            else:
                chart_cfg = {"enabled": False}
            settings_copy.backtest["chart"] = chart_cfg
            try:
                _, _, _, result, data = run_backtest_with_result(settings_copy)
                stats = result.stats()
                with tempfile.NamedTemporaryFile(
                    prefix="comp_equity_price_mdd_", suffix=".png", delete=False
                ) as tmp_file:
                    output_path = Path(tmp_file.name)
                title = f"{var1['label']}={value1}"
                chart_path = create_equity_price_mdd_chart(
                    result,
                    data,
                    output_path,
                    title=title,
                )
                results.append(
                    {
                        "value": value1,
                        "stats": stats,
                        "chart_path": chart_path,
                    }
                )
            except Exception as exc:
                results.append({"value": value1, "error": f"{type(exc).__name__}"})
            completed += 1
            if progress_cb:
                progress_cb(completed, total)
        return results

    def _run_costom3_backtests(
        self,
        base_settings: Settings,
        var1: dict,
        values1: List[Any],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        total_runs: Optional[int] = None,
    ) -> List[dict]:
        results: List[dict] = []
        total = total_runs or len(values1)
        completed = 0
        if progress_cb:
            progress_cb(0, total)
        for value1 in values1:
            settings_copy = copy.deepcopy(base_settings)
            applied = self._apply_setting_value_to(
                settings_copy,
                var1["section"],
                var1["key"],
                value1,
                var1["index"],
            )
            if not applied:
                results.append({"value": value1, "error": "Failed to apply setting"})
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)
                continue
            chart_cfg = settings_copy.backtest.get("chart")
            if isinstance(chart_cfg, dict):
                chart_cfg = dict(chart_cfg)
                chart_cfg["enabled"] = False
            else:
                chart_cfg = {"enabled": False}
            settings_copy.backtest["chart"] = chart_cfg
            try:
                _, _, _, result, _ = run_backtest_with_result(settings_copy)
                results.append(
                    {
                        "value": value1,
                        "result": result,
                        "stats": result.stats(),
                    }
                )
            except Exception as exc:
                results.append({"value": value1, "error": f"{type(exc).__name__}"})
            completed += 1
            if progress_cb:
                progress_cb(completed, total)
        return results

    @staticmethod
    def _format_comp_output(
        var1: dict,
        value: Any,
        stats: Dict[str, Any],
        data_label: Optional[str] = None,
    ) -> str:
        def fmt_value(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)

        def fmt_pct(value: Any) -> str:
            return f"{float(value) * 100.0:.2f}%"

        def fmt_pf(value: Any) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return "n/a"
            if not np.isfinite(val):
                return "inf"
            return f"{val:.2f}"

        total_return = fmt_pct(stats.get("total_return", 0.0))
        max_drawdown = fmt_pct(stats.get("max_drawdown", 0.0))
        win_rate = fmt_pct(stats.get("win_rate", 0.0))
        profit_factor = fmt_pf(stats.get("profit_factor", 0.0))
        total_trades = int(stats.get("total_trades", 0))

        lines = ["**🧪 Comp Result**"]
        if data_label:
            lines.append(f"🗂️  Data: `{data_label}`")
        lines.append(f"🧩 Var: `{var1['label']}={fmt_value(value)}`")
        lines.append(
            f"📈 Total return: {total_return} | Max DD: {max_drawdown} | "
            f"Profit factor: {profit_factor} | Win rate: {win_rate} | Trades: {total_trades}"
        )
        return "\n".join(lines)

    def _format_grid_output(
        self,
        var1: dict,
        values1: List[Any],
        var2: Optional[dict],
        values2: Optional[List[Any]],
        results: dict,
        data_label: Optional[str] = None,
    ) -> str:
        def fmt_value(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.4g}"
            return str(value)

        def fmt_pct(value: Any) -> str:
            return f"{float(value) * 100.0:.2f}%"

        def fmt_pct_signed(value: Any) -> str:
            return f"{float(value) * 100.0:+.2f}%"

        lines: List[str] = ["**📊 Grid Summary**", "━━━━━━━━━━━━━━━━━━"]
        if data_label:
            lines.append(f"🗂️  Data: `{data_label}`")
        header = f"`{var1['label']}`"
        if var2:
            header += f" x `{var2['label']}`"
        lines.append(f"🧩 Variables: {header}")

        total_runs = len(values1) * (len(values2) if values2 else 1)
        ok_runs = 0
        err_runs = 0
        for value1 in values1:
            row = results["rows"].get(value1, {})
            run_values2 = values2 or [None]
            for value2 in run_values2:
                cell = row.get(value2, {})
                if "stats" in cell:
                    ok_runs += 1
                else:
                    err_runs += 1
        lines.append(f"🔁 Runs: {total_runs} total | ✅ {ok_runs} | ❌ {err_runs}")

        best = results.get("best")
        if best:
            stats = best["stats"]
            params = f"`{var1['label']}={fmt_value(best['value1'])}`"
            if var2:
                params += f", `{var2['label']}={fmt_value(best['value2'])}`"
            lines.append("━━━━━━━━━━━━━━━━━━")
            lines.append("**🥇 Best**")
            lines.append(f"• Params: {params}")
            lines.append(
                f"• Return: {fmt_pct_signed(stats.get('total_return', 0.0))} | "
                f"DD: {fmt_pct(stats.get('max_drawdown', 0.0))} | "
                f"WR: {fmt_pct(stats.get('win_rate', 0.0))} | "
                f"Trades: {int(stats.get('total_trades', 0))}"
            )
        worst = results.get("worst")
        if worst and worst is not best:
            stats = worst["stats"]
            params = f"`{var1['label']}={fmt_value(worst['value1'])}`"
            if var2:
                params += f", `{var2['label']}={fmt_value(worst['value2'])}`"
            lines.append("━━━━━━━━━━━━━━━━━━")
            lines.append("**💥 Worst**")
            lines.append(f"• Params: {params}")
            lines.append(
                f"• Return: {fmt_pct_signed(stats.get('total_return', 0.0))} | "
                f"DD: {fmt_pct(stats.get('max_drawdown', 0.0))} | "
                f"WR: {fmt_pct(stats.get('win_rate', 0.0))} | "
                f"Trades: {int(stats.get('total_trades', 0))}"
            )
        return "\n".join(lines)

    def _build_grid_heatmap_images(
        self,
        var1: dict,
        values1: List[Any],
        var2: Optional[dict],
        values2: Optional[List[Any]],
        results: dict,
        data_label: Optional[str] = None,
    ) -> List[Path]:
        if not values1:
            return []

        if var2 is None or not values2:
            values2_used = [""]
            label2 = ""
            title_base = f"Grid heatmap: {var1['label']}"
        else:
            values2_used = values2
            label2 = var2["label"]
            title_base = f"Grid heatmap: {var1['label']} x {var2['label']}"

        if data_label:
            title_base = f"{title_base} ({data_label})"

        def chunk_indices(total: int, max_size: int) -> List[tuple[int, int]]:
            if total <= 0:
                return []
            ranges: List[tuple[int, int]] = []
            for start in range(0, total, max_size):
                end = min(total, start + max_size)
                ranges.append((start, end))
            return ranges

        def build_matrix(metric_key: str, scale: float = 1.0) -> List[List[Optional[float]]]:
            matrix: List[List[Optional[float]]] = []
            for value1 in values1:
                row: List[Optional[float]] = []
                for value2 in values2_used:
                    cell_key = None if var2 is None or not values2 else value2
                    cell = results["rows"][value1].get(cell_key, {})
                    if "stats" not in cell:
                        row.append(None)
                    else:
                        row.append(float(cell["stats"].get(metric_key, 0.0)) * scale)
                matrix.append(row)
            return matrix

        metrics = [
            ("total_return", 100.0, "Return %", "return", None),
            ("max_drawdown", 100.0, "Max DD %", "mdd", lambda value: f"{value:.2f}%"),
            ("win_rate", 100.0, "Win rate %", "winrate", lambda value: f"{value:.2f}%"),
        ]

        max_rows = 20
        max_cols = 20
        row_chunks = chunk_indices(len(values1), max_rows)
        col_chunks = chunk_indices(len(values2_used), max_cols)
        show_values = True
        images: List[Path] = []
        for metric_key, scale, label, suffix, formatter in metrics:
            matrix = build_matrix(metric_key, scale)
            flat_values = [
                float(value)
                for row in matrix
                for value in row
                if value is not None and np.isfinite(value)
            ]
            max_abs = max((abs(value) for value in flat_values), default=0.0)
            for row_start, row_end in row_chunks:
                for col_start, col_end in col_chunks:
                    sub_values1 = values1[row_start:row_end]
                    sub_values2 = values2_used[col_start:col_end]
                    sub_matrix = [
                        row[col_start:col_end] for row in matrix[row_start:row_end]
                    ]
                    chunk_note = ""
                    if len(row_chunks) > 1 or len(col_chunks) > 1:
                        if label2:
                            chunk_note = (
                                f" (rows {row_start + 1}-{row_end}/{len(values1)}, "
                                f"cols {col_start + 1}-{col_end}/{len(values2_used)})"
                            )
                        else:
                            chunk_note = f" (rows {row_start + 1}-{row_end}/{len(values1)})"
                    with tempfile.NamedTemporaryFile(
                        prefix=f"grid_heatmap_{suffix}_", suffix=".png", delete=False
                    ) as tmp_file:
                        path = Path(tmp_file.name)
                    title = f"{title_base} - {label}{chunk_note}"
                    created = create_grid_heatmap(
                        values1=sub_values1,
                        values2=sub_values2,
                        matrix=sub_matrix,
                        output_path=path,
                        title=title,
                        label1=var1["label"],
                        label2=label2,
                        show_values=show_values,
                        colorbar_label=label,
                        value_formatter=formatter,
                        max_abs=max_abs,
                    )
                    if created:
                        images.append(created)
        return images

    def _build_grid_data_file(
        self,
        var1: dict,
        values1: List[Any],
        var2: Optional[dict],
        values2: Optional[List[Any]],
        results: dict,
        data_label: Optional[str] = None,
    ) -> List[Path]:
        if not values1:
            return []
        run_values2 = values2 or [None]
        columns: List[str] = []
        if data_label:
            columns.append("data")
        columns.append(var1["label"])
        if var2:
            columns.append(var2["label"])
        columns.extend(
            [
                "total_return_pct",
                "max_drawdown_pct",
                "win_rate_pct",
                "total_trades",
                "profit_factor",
                "avg_return_pct",
                "avg_trade",
                "avg_win",
                "avg_loss",
                "largest_win",
                "largest_loss",
                "error",
            ]
        )

        with tempfile.NamedTemporaryFile(
            prefix="grid_results_", suffix=".csv", delete=False, mode="w", newline=""
        ) as tmp_file:
            writer = csv.writer(tmp_file)
            writer.writerow(columns)
            for value1 in values1:
                row = results["rows"].get(value1, {})
                for value2 in run_values2:
                    cell = row.get(value2, {})
                    stats = cell.get("stats")
                    error = cell.get("error")
                    line: List[Any] = []
                    if data_label:
                        line.append(data_label)
                    line.append(value1)
                    if var2:
                        line.append(value2)
                    if stats:
                        line.extend(
                            [
                                float(stats.get("total_return", 0.0)) * 100.0,
                                float(stats.get("max_drawdown", 0.0)) * 100.0,
                                float(stats.get("win_rate", 0.0)) * 100.0,
                                int(stats.get("total_trades", 0)),
                                stats.get("profit_factor", 0.0),
                                float(stats.get("avg_return", 0.0)) * 100.0,
                                stats.get("avg_trade", 0.0),
                                stats.get("avg_win", 0.0),
                                stats.get("avg_loss", 0.0),
                                stats.get("largest_win", 0.0),
                                stats.get("largest_loss", 0.0),
                                "",
                            ]
                        )
                    else:
                        line.extend([""] * 11)
                        line.append(error or "missing")
                    writer.writerow(line)
        return [Path(tmp_file.name)]

    async def _send_grid_output(
        self,
        send_func,
        content: str,
        image_paths: Optional[Sequence[Path]] = None,
        data_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        files: List[discord.File] = []
        for path in image_paths or []:
            files.append(discord.File(path))
        for path in data_paths or []:
            files.append(discord.File(path))
        if len(content) > 1900:
            content = content[:1860].rstrip() + " ... (truncated)"
        await send_func(content=content, files=files or None)

    def _parse_bulk_updates(self, updates: str) -> tuple[List[tuple[str, str, Any]], List[str]]:
        results: List[tuple[str, str, Any]] = []
        errors: List[str] = []
        allowed_sections = {"data", "backtest", "strategy"}
        lines = [line.strip() for line in updates.splitlines() if line.strip()]
        for idx, line in enumerate(lines, start=1):
            if "=" not in line:
                errors.append(f"Line {idx}: missing '=' in '{line}'.")
                continue
            left, raw_value = line.split("=", 1)
            left = left.strip()
            raw_value = raw_value.strip()
            if "." not in left:
                errors.append(f"Line {idx}: use section.key format in '{left}'.")
                continue
            section, key = left.split(".", 1)
            section = section.strip()
            key = key.strip()
            if section not in allowed_sections:
                errors.append(f"Line {idx}: invalid section '{section}'.")
                continue
            if not key:
                errors.append(f"Line {idx}: missing key after section.")
                continue
            value = self._coerce_value(raw_value)
            results.append((section, key, value))
        return results, errors

    def _flatten_settings(self) -> List[tuple[str, Any]]:
        entries: List[tuple[str, Any]] = []
        for section in ("data", "backtest", "strategy"):
            data = getattr(self.settings, section)
            entries.extend(self._flatten_section(section, data))
        return entries

    def _grid_variable_keys(self) -> List[str]:
        keys: List[str] = []
        for full_key, value in self._flatten_settings():
            if isinstance(value, (int, float)):
                keys.append(full_key)
        return sorted(set(keys))

    @staticmethod
    def _parse_costom_params(
        lookback_value: str,
        buffer_value: str,
    ) -> tuple[Optional[int], Optional[float], Optional[str]]:
        try:
            lookback = int(float(str(lookback_value).strip()))
        except (TypeError, ValueError):
            return None, None, "Lookback must be an integer."
        if lookback <= 0:
            return None, None, "Lookback must be positive."

        raw = str(buffer_value).strip().replace("%", "")
        try:
            buffer_pct = abs(float(raw)) / 100.0
        except (TypeError, ValueError):
            return None, None, "Buffer must be a number (percent)."
        if buffer_pct < 0:
            buffer_pct = abs(buffer_pct)
        return lookback, buffer_pct, None

    @staticmethod
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

    @staticmethod
    def _compute_costom_exit_reason_stats(
        result,
        data: pd.DataFrame,
        lookback: int,
        buffer_pct: float,
        settings: Settings,
    ) -> tuple[Dict[str, Dict[str, Any]], int, int]:
        exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
        required_cols = {"high", "low"}
        if data.empty or not required_cols.issubset(data.columns):
            return {}, 0, len(exit_trades)

        reason_stats: Dict[str, Dict[str, Any]] = {}
        evaluated = 0
        skipped = 0
        for trade in exit_trades:
            reason = str(trade.get("exit_reason") or "unknown")
            if reason == "end_of_data":
                skipped += 1
                continue
            entry_time = trade.get("entry_time")
            entry_idx = BacktestBot._resolve_trade_index(data, entry_time)
            if entry_idx is None or entry_idx - lookback < 0:
                skipped += 1
                continue
            exit_time = trade.get("exit_time") or trade.get("timestamp")
            exit_idx = BacktestBot._resolve_trade_index(data, exit_time)
            if exit_idx is None or exit_idx < entry_idx:
                skipped += 1
                continue
            entry_row = data.iloc[entry_idx]
            if pd.isna(entry_row.get("low")) or pd.isna(entry_row.get("high")):
                skipped += 1
                continue
            window = data.iloc[entry_idx - lookback : entry_idx]
            if window.empty:
                skipped += 1
                continue
            trade_window = data.iloc[entry_idx : exit_idx + 1]
            if trade_window.empty:
                skipped += 1
                continue
            direction = int(trade.get("direction", 0))
            def _coerce_price(value: Any) -> Optional[float]:
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(val) or val <= 0:
                    return None
                return val

            entry_price = _coerce_price(trade.get("entry_price"))
            if entry_price is None:
                entry_price = _coerce_price(entry_row.get("open"))
            if entry_price is None:
                entry_price = _coerce_price(entry_row.get("close"))
            if entry_price is None:
                skipped += 1
                continue
            leverage = BacktestBot._resolve_costom_leverage(settings, direction)
            if direction == 1:
                prior_low = float(window["low"].min())
                threshold = prior_low * (1.0 - buffer_pct)
                breached = float(trade_window["low"].min()) <= threshold
                loss_pct = ((threshold / entry_price) - 1.0) * leverage * 100.0
            elif direction == -1:
                prior_high = float(window["high"].max())
                threshold = prior_high * (1.0 + buffer_pct)
                breached = float(trade_window["high"].max()) >= threshold
                loss_pct = ((entry_price - threshold) / entry_price) * leverage * 100.0
            else:
                skipped += 1
                continue
            evaluated += 1
            stats = reason_stats.setdefault(reason, {"hit": 0, "total": 0, "losses": []})
            stats["total"] += 1
            if breached:
                stats["hit"] += 1
                stats["losses"].append(float(loss_pct))

        return reason_stats, evaluated, skipped

    @staticmethod
    def _resolve_costom_leverage(settings: Settings, direction: int) -> float:
        base = settings.backtest.get("leverage", 1.0)
        try:
            base_val = float(base)
        except (TypeError, ValueError):
            base_val = 1.0
        if base_val <= 0:
            base_val = 1.0

        params = settings.strategy.get("params") or {}
        strategy_name = str(settings.strategy.get("name") or "").lower()
        if strategy_name == "cheatkey":
            params = _normalize_cheatkey_params(params)

        def _get_param(key: str) -> Optional[float]:
            if key in params:
                value = params.get(key)
            else:
                value = params.get(key.lower())
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if value <= 0:
                return None
            return value

        if direction > 0:
            resolved = _get_param("long_leverage")
        else:
            resolved = _get_param("short_leverage")
        return resolved if resolved is not None else base_val

    @staticmethod
    @staticmethod
    def _format_costom_output(
        data_label: str,
        lookback: int,
        buffer_pct: float,
        reason_stats: Dict[str, Dict[str, Any]],
        evaluated: int,
        skipped: int,
    ) -> str:
        buffer_text = f"{buffer_pct * 100.0:.3g}%"
        lines = [
            "**🧭 Costom pre-low/high stats**",
            f"🗂️  Data: `{data_label}`",
            f"🔍 Lookback: {lookback} bars | Buffer: {buffer_text}",
            f"✅ Evaluated: {evaluated} | ⏭️ Skipped: {skipped}",
            "━━━━━━━━━━━━━━━━━━",
            "Exit reason → break ratio",
        ]
        if not reason_stats:
            lines.append("No exit trades to evaluate.")
            return "\n".join(lines)

        for reason, stats in sorted(
            reason_stats.items(), key=lambda item: item[1].get("total", 0), reverse=True
        ):
            total = int(stats.get("total", 0))
            hit = int(stats.get("hit", 0))
            ratio = (hit / total * 100.0) if total else 0.0
            lines.append(f"- {reason}: {hit}/{total} ({ratio:.1f}%)")
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("Breach loss% stats (leveraged, mean/median)")
        for reason, stats in sorted(
            reason_stats.items(), key=lambda item: item[1].get("total", 0), reverse=True
        ):
            losses = stats.get("losses", [])
            if not losses:
                lines.append(f"- {reason}: n/a")
                continue
            mean_val = float(np.mean(losses))
            median_val = float(np.median(losses))
            lines.append(f"- {reason}: mean {mean_val:.2f}% | median {median_val:.2f}%")
        lines.append(
            "Note: long checks prior low - buffer; short checks prior high + buffer, "
            "across the full trade window."
        )
        lines.append("Note: end_of_data exits are excluded from the stats.")
        lines.append(
            "Loss% uses leverage from strategy long/short leverage when available, "
            "otherwise backtest leverage."
        )
        return "\n".join(lines)

    @staticmethod
    def _compute_costom_bundle(
        result,
        data: pd.DataFrame,
        lookback: int,
        buffer_pct: float,
        settings: Settings,
        data_label: str,
    ) -> tuple[str, List[Path], Dict[str, Dict[str, Any]], int, int]:
        reason_stats, evaluated, skipped = BacktestBot._compute_costom_exit_reason_stats(
            result,
            data,
            lookback,
            buffer_pct,
            settings,
        )
        message = BacktestBot._format_costom_output(
            data_label=data_label,
            lookback=lookback,
            buffer_pct=buffer_pct,
            reason_stats=reason_stats,
            evaluated=evaluated,
            skipped=skipped,
        )
        chart_paths = BacktestBot._build_costom_loss_charts(reason_stats=reason_stats)
        return message, chart_paths, reason_stats, evaluated, skipped

    @staticmethod
    def _build_costom_loss_charts(
        reason_stats: Dict[str, Dict[str, Any]],
    ) -> List[Path]:
        chart_paths: List[Path] = []
        for reason, stats in sorted(
            reason_stats.items(), key=lambda item: item[1].get("total", 0), reverse=True
        ):
            losses = stats.get("losses", [])
            if not losses:
                continue
            safe_reason = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in reason)
            with tempfile.NamedTemporaryFile(
                prefix=f"costom_{safe_reason}_",
                suffix=".png",
                delete=False,
            ) as tmp_file:
                output_path = Path(tmp_file.name)
            chart_path = create_costom_loss_chart(
                reason=reason,
                losses=losses,
                output_path=output_path,
            )
            if chart_path:
                chart_paths.append(chart_path)
        return chart_paths

    @staticmethod
    def _format_costom4_output(
        data_label: str,
        monthly_df: pd.DataFrame,
        stats: Dict[str, Any],
        entry_minute_df: pd.DataFrame,
        entry_minute_stats: Dict[str, Any],
        entry_minute_tp_df: pd.DataFrame,
        entry_minute_tp_stats: Dict[str, Any],
    ) -> str:
        months = int(stats.get("months", 0) or 0)
        lines = [
            "**📅 Costom4 monthly return stability**",
            f"🗂️  Data: `{data_label}`",
            f"📈 Months: {months}",
            "━━━━━━━━━━━━━━━━━━",
            "Stability stats (lower dispersion = more stable)",
        ]
        if months <= 0:
            lines.append("- No monthly returns available.")
            monthly_output = "\n".join(lines)
            entry_lines = [
                "━━━━━━━━━━━━━━━━━━",
                "Entry minute win rate (by entry time)",
            ]
            if entry_minute_df.empty:
                entry_lines.append("- n/a")
            else:
                total_trades = int(entry_minute_stats.get("trades", 0) or 0)
                overall_wr = float(entry_minute_stats.get("win_rate", 0.0) or 0.0)
                entry_lines.append(f"- Trades: {total_trades} | Win rate: {overall_wr:.1f}%")
                for _, row in entry_minute_df.iterrows():
                    label = str(row.get("label", "")).zfill(2)
                    win_rate = float(row.get("win_rate", 0.0))
                    wins = int(row.get("wins", 0))
                    total = int(row.get("total", 0))
                    entry_lines.append(f"- {label}: {win_rate:.1f}% (W {wins} / T {total})")
            entry_lines.append("━━━━━━━━━━━━━━━━━━")
            entry_lines.append("Entry minute TP reach rate (by entry time)")
            if entry_minute_tp_df.empty:
                total_trades = int(entry_minute_tp_stats.get("trades", 0) or 0)
                overall_tp = float(entry_minute_tp_stats.get("tp_rate", 0.0) or 0.0)
                covered = int(entry_minute_tp_stats.get("trades_with_time", total_trades) or 0)
                if total_trades > 0:
                    entry_lines.append(
                        f"- Trades: {total_trades} (minute-covered {covered}) | TP rate: {overall_tp:.1f}%"
                    )
                    entry_lines.append("- minute breakdown: n/a")
                else:
                    entry_lines.append("- n/a")
            else:
                total_trades = int(entry_minute_tp_stats.get("trades", 0) or 0)
                overall_tp = float(entry_minute_tp_stats.get("tp_rate", 0.0) or 0.0)
                covered = int(entry_minute_tp_stats.get("trades_with_time", total_trades) or 0)
                entry_lines.append(
                    f"- Trades: {total_trades} (minute-covered {covered}) | TP rate: {overall_tp:.1f}%"
                )
                for _, row in entry_minute_tp_df.iterrows():
                    label = str(row.get("label", "")).zfill(2)
                    tp_rate = float(row.get("tp_rate", 0.0))
                    tp_hits = int(row.get("tp_hits", 0))
                    total = int(row.get("total", 0))
                    entry_lines.append(f"- {label}: {tp_rate:.1f}% (TP {tp_hits} / T {total})")
            return "\n".join([monthly_output, *entry_lines])

        mean = stats.get("mean", 0.0)
        median = stats.get("median", 0.0)
        std = stats.get("std", 0.0)
        mad = stats.get("mad", 0.0)
        iqr = stats.get("iqr", 0.0)
        min_val = stats.get("min", 0.0)
        max_val = stats.get("max", 0.0)
        pos_rate = stats.get("pos_rate", 0.0)
        cv = stats.get("cv", np.nan)

        lines += [
            f"- Mean: {mean:.2f}% | Median: {median:.2f}%",
            f"- Std: {std:.2f}% | MAD: {mad:.2f}%",
            f"- IQR: {iqr:.2f}% | Min/Max: {min_val:.2f}% / {max_val:.2f}%",
            f"- +Rate: {pos_rate:.1f}%",
        ]
        if np.isfinite(cv):
            lines.append(f"- CV: {cv:.2f}")

        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("Monthly returns")
        if monthly_df.empty:
            lines.append("- n/a")
            monthly_output = "\n".join(lines)
        else:
            for _, row in monthly_df.iterrows():
                month = str(row.get("month", ""))
                value = row.get("return_pct", np.nan)
                if value is None or not np.isfinite(value):
                    lines.append(f"- {month}: n/a")
                else:
                    lines.append(f"- {month}: {float(value):.2f}%")
            monthly_output = "\n".join(lines)

        entry_lines = [
            "━━━━━━━━━━━━━━━━━━",
            "Entry minute win rate (by entry time)",
        ]
        if entry_minute_df.empty:
            entry_lines.append("- n/a")
        else:
            total_trades = int(entry_minute_stats.get("trades", 0) or 0)
            overall_wr = float(entry_minute_stats.get("win_rate", 0.0) or 0.0)
            entry_lines.append(f"- Trades: {total_trades} | Win rate: {overall_wr:.1f}%")
            for _, row in entry_minute_df.iterrows():
                label = str(row.get("label", "")).zfill(2)
                win_rate = float(row.get("win_rate", 0.0))
                wins = int(row.get("wins", 0))
                total = int(row.get("total", 0))
                entry_lines.append(f"- {label}: {win_rate:.1f}% (W {wins} / T {total})")
        entry_lines.append("━━━━━━━━━━━━━━━━━━")
        entry_lines.append("Entry minute TP reach rate (by entry time)")
        if entry_minute_tp_df.empty:
            total_trades = int(entry_minute_tp_stats.get("trades", 0) or 0)
            overall_tp = float(entry_minute_tp_stats.get("tp_rate", 0.0) or 0.0)
            covered = int(entry_minute_tp_stats.get("trades_with_time", total_trades) or 0)
            if total_trades > 0:
                entry_lines.append(
                    f"- Trades: {total_trades} (minute-covered {covered}) | TP rate: {overall_tp:.1f}%"
                )
                entry_lines.append("- minute breakdown: n/a")
            else:
                entry_lines.append("- n/a")
        else:
            total_trades = int(entry_minute_tp_stats.get("trades", 0) or 0)
            overall_tp = float(entry_minute_tp_stats.get("tp_rate", 0.0) or 0.0)
            covered = int(entry_minute_tp_stats.get("trades_with_time", total_trades) or 0)
            entry_lines.append(
                f"- Trades: {total_trades} (minute-covered {covered}) | TP rate: {overall_tp:.1f}%"
            )
            for _, row in entry_minute_tp_df.iterrows():
                label = str(row.get("label", "")).zfill(2)
                tp_rate = float(row.get("tp_rate", 0.0))
                tp_hits = int(row.get("tp_hits", 0))
                total = int(row.get("total", 0))
                entry_lines.append(f"- {label}: {tp_rate:.1f}% (TP {tp_hits} / T {total})")

        return "\n".join([monthly_output, *entry_lines])

    async def _run_grid_from_interaction(
        self,
        interaction: discord.Interaction,
        tokens: List[str],
        data_file: Optional[str],
    ) -> None:
        if len(tokens) not in {4, 8}:
            await interaction.response.send_message(
                "Usage: `!grid [--data FILE] section.key start end step "
                "[section.key start end step]`",
                ephemeral=True,
            )
            return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, interaction.channel)
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await interaction.response.send_message(
                        f"Data file not found: {data_file} (use `!data list`)",
                        ephemeral=True,
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        var1, err = self._parse_grid_var(base_settings, tokens[0])
        if err:
            await interaction.response.send_message(
                f"Invalid first variable: {err}",
                ephemeral=True,
            )
            return
        values1, err = self._build_grid_values(tokens[1], tokens[2], tokens[3])
        if err:
            await interaction.response.send_message(
                f"Invalid first range: {err}",
                ephemeral=True,
            )
            return

        var2: Optional[dict] = None
        values2: Optional[List[Any]] = None
        if len(tokens) == 8:
            var2, err = self._parse_grid_var(base_settings, tokens[4])
            if err:
                await interaction.response.send_message(
                    f"Invalid second variable: {err}",
                    ephemeral=True,
                )
                return
            values2, err = self._build_grid_values(tokens[5], tokens[6], tokens[7])
            if err:
                await interaction.response.send_message(
                    f"Invalid second range: {err}",
                    ephemeral=True,
                )
                return

        max_runs = 250
        total_runs = len(values1) * (len(values2) if values2 else 1)
        if total_runs > max_runs:
            await interaction.response.send_message(
                f"Too many combinations ({total_runs}). Please keep it under {max_runs}.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        try:
            progress_msg = await interaction.followup.send(
                f"Running grid backtests (0/{total_runs})...",
                wait=True,
            )
        except discord.HTTPException as exc:
            logger.warning("Grid followup send failed; falling back to channel: %s", exc)
            if interaction.channel is None:
                raise
            progress_msg = await interaction.channel.send(
                f"Running grid backtests (0/{total_runs})..."
            )
        progress_channel = progress_msg.channel or interaction.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(interaction.channel.id)
        )
        progress_handle = self._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running grid backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Grid progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Grid progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running grid backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Grid progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self._run_grid_backtests,
                base_settings,
                var1,
                values1,
                var2,
                values2,
                progress_cb,
                total_runs,
            )
        finally:
            self._unregister_grid_progress(progress_handle)
        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        output = self._format_grid_output(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )
        image_paths = self._build_grid_heatmap_images(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )
        data_paths = self._build_grid_data_file(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or interaction.channel
        if target_channel is None:
            await interaction.followup.send(
                "Grid backtest complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return
        sent = await self._send_grid_output_to_channel(
            target_channel.id,
            output,
            image_paths=image_paths,
            data_paths=data_paths,
        )
        if not sent:
            self._queue_grid_output(
                target_channel.id,
                output,
                image_paths=image_paths,
                data_paths=data_paths,
            )
        if result_channel is not None:
            await interaction.followup.send(
                f"Grid backtest complete. Results posted in #{result_channel.name}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Grid backtest complete. Results posted here.",
                ephemeral=True,
            )

    async def _run_costom_from_interaction(
        self,
        interaction: discord.Interaction,
        tokens: List[str],
        data_file: Optional[str],
    ) -> None:
        if len(tokens) != 2:
            await interaction.response.send_message(
                "Usage: `!costom [--data FILE] lookback buffer_percent`\n"
                "Example: `!costom --data BTC.csv 20 0.1`",
                ephemeral=True,
            )
            return

        lookback, buffer_pct, err = self._parse_costom_params(tokens[0], tokens[1])
        if err:
            await interaction.response.send_message(err, ephemeral=True)
            return

        base_settings = load_settings(self.settings_path)
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await interaction.response.send_message(
                        f"Data file not found: {data_file} (use `!data list`)",
                        ephemeral=True,
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        await interaction.response.defer(thinking=True)
        loop = asyncio.get_running_loop()
        settings_copy = copy.deepcopy(base_settings)
        chart_cfg = settings_copy.backtest.get("chart")
        if isinstance(chart_cfg, dict):
            chart_cfg = dict(chart_cfg)
            chart_cfg["enabled"] = False
        else:
            chart_cfg = {"enabled": False}
        settings_copy.backtest["chart"] = chart_cfg

        try:
            _, _, _, result, data = await loop.run_in_executor(
                None,
                run_backtest_with_result,
                settings_copy,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Costom failed: {exc}",
                ephemeral=True,
            )
            return

        self._last_backtest = {"result": result, "data": data}
        data_pattern = settings_copy.data.get("file_pattern", "*.csv")
        message, chart_paths, _, _, _ = await asyncio.to_thread(
            BacktestBot._compute_costom_bundle,
            result,
            data,
            lookback,
            buffer_pct,
            settings_copy,
            str(data_pattern),
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or interaction.channel
        if target_channel is None:
            await interaction.followup.send(
                "Costom complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )
        if result_channel is not None:
            await interaction.followup.send(
                f"Costom complete. Results posted in #{result_channel.name}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Costom complete. Results posted here.",
                ephemeral=True,
            )

    async def _run_costom2_from_interaction(
        self,
        interaction: discord.Interaction,
        tokens: List[str],
        data_file: Optional[str],
    ) -> None:
        if tokens:
            await interaction.response.send_message(
                "Usage: `!costom2 [--data FILE]`",
                ephemeral=True,
            )
            return

        base_settings = load_settings(self.settings_path)
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await interaction.response.send_message(
                        f"Data file not found: {data_file} (use `!data list`)",
                        ephemeral=True,
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        await interaction.response.defer(thinking=True)
        loop = asyncio.get_running_loop()
        settings_copy = copy.deepcopy(base_settings)
        chart_cfg = settings_copy.backtest.get("chart")
        if isinstance(chart_cfg, dict):
            chart_cfg = dict(chart_cfg)
            chart_cfg["enabled"] = False
        else:
            chart_cfg = {"enabled": False}
        settings_copy.backtest["chart"] = chart_cfg

        try:
            _, _, _, result, data = await loop.run_in_executor(
                None,
                run_backtest_with_result,
                settings_copy,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Costom2 failed: {exc}",
                ephemeral=True,
            )
            return

        self._last_backtest = {"result": result, "data": None}
        data_pattern = settings_copy.data.get("file_pattern", "*.csv")
        with tempfile.NamedTemporaryFile(
            prefix="costom2_monthly_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            output_path = Path(tmp_file.name)
        chart_path = await asyncio.to_thread(
            create_equity_mdd_monthly_stats_chart,
            result,
            output_path,
            title=f"Equity + MDD + Monthly stats ({data_pattern})",
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or interaction.channel
        if target_channel is None:
            await interaction.followup.send(
                "Costom2 complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return

        message = f"**📊 Costom2 monthly stats**\n🗂️  Data: `{data_pattern}`"
        chart_paths = [chart_path] if chart_path else None
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )

        if result_channel is not None:
            await interaction.followup.send(
                f"Costom2 complete. Results posted in #{result_channel.name}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Costom2 complete. Results posted here.",
                ephemeral=True,
            )

    async def _run_costom4_from_interaction(
        self,
        interaction: discord.Interaction,
        tokens: List[str],
        data_file: Optional[str],
    ) -> None:
        if tokens:
            await interaction.response.send_message(
                "Usage: `!costom4 [--data FILE]`",
                ephemeral=True,
            )
            return

        base_settings = load_settings(self.settings_path)
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await interaction.response.send_message(
                        f"Data file not found: {data_file} (use `!data list`)",
                        ephemeral=True,
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        await interaction.response.defer(thinking=True)
        loop = asyncio.get_running_loop()
        settings_copy = copy.deepcopy(base_settings)
        chart_cfg = settings_copy.backtest.get("chart")
        if isinstance(chart_cfg, dict):
            chart_cfg = dict(chart_cfg)
            chart_cfg["enabled"] = False
        else:
            chart_cfg = {"enabled": False}
        settings_copy.backtest["chart"] = chart_cfg

        try:
            _, _, _, result, data = await loop.run_in_executor(
                None,
                run_backtest_with_result,
                settings_copy,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Costom4 failed: {exc}",
                ephemeral=True,
            )
            return

        self._last_backtest = {"result": result, "data": data}
        data_pattern = settings_copy.data.get("file_pattern", "*.csv")
        monthly_df, stats = compute_monthly_return_stats(result)
        entry_minute_df, entry_minute_stats = compute_entry_minute_win_rate(result)
        entry_minute_tp_df, entry_minute_tp_stats = compute_entry_minute_tp_rate(result)
        message = BacktestBot._format_costom4_output(
            data_label=str(data_pattern),
            monthly_df=monthly_df,
            stats=stats,
            entry_minute_df=entry_minute_df,
            entry_minute_stats=entry_minute_stats,
            entry_minute_tp_df=entry_minute_tp_df,
            entry_minute_tp_stats=entry_minute_tp_stats,
        )

        with tempfile.NamedTemporaryFile(
            prefix="costom4_monthly_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            output_path = Path(tmp_file.name)
        chart_path = await asyncio.to_thread(
            create_monthly_return_stability_chart,
            result,
            output_path,
            title=f"Monthly return stability ({data_pattern})",
        )
        with tempfile.NamedTemporaryFile(
            prefix="costom4_entry_minute_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            minute_output_path = Path(tmp_file.name)
        entry_minute_chart = await asyncio.to_thread(
            create_entry_minute_win_rate_chart,
            result,
            minute_output_path,
            title=f"Entry minute win rate ({data_pattern})",
        )
        with tempfile.NamedTemporaryFile(
            prefix="costom4_entry_minute_tp_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            tp_output_path = Path(tmp_file.name)
        entry_minute_tp_chart = await asyncio.to_thread(
            create_entry_minute_tp_rate_chart,
            result,
            tp_output_path,
            title=f"Entry minute TP reach rate ({data_pattern})",
        )
        equity_price_chart = None
        if isinstance(data, pd.DataFrame) and not data.empty:
            with tempfile.NamedTemporaryFile(
                prefix="costom4_equity_price_",
                suffix=".png",
                delete=False,
            ) as tmp_file:
                price_output_path = Path(tmp_file.name)
            equity_price_chart = await asyncio.to_thread(
                create_equity_price_chart,
                result,
                data,
                price_output_path,
            )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or interaction.channel
        if target_channel is None:
            await interaction.followup.send(
                "Costom4 complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return

        chart_paths = []
        if chart_path:
            chart_paths.append(chart_path)
        if entry_minute_chart:
            chart_paths.append(entry_minute_chart)
        if entry_minute_tp_chart:
            chart_paths.append(entry_minute_tp_chart)
        if equity_price_chart:
            chart_paths.append(equity_price_chart)
        if not chart_paths:
            chart_paths = None
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )

        if result_channel is not None:
            await interaction.followup.send(
                f"Costom4 complete. Results posted in #{result_channel.name}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Costom4 complete. Results posted here.",
                ephemeral=True,
            )

    async def _run_comp_from_interaction(
        self,
        interaction: discord.Interaction,
        tokens: List[str],
        data_file: Optional[str],
    ) -> None:
        if len(tokens) != 4:
            await interaction.response.send_message(
                "Usage: `!comp [--data FILE] section.key start end step`\n"
                "Example: `!comp --data BTC.csv strategy.params.fast_window 5 20 1`",
                ephemeral=True,
            )
            return
        base_settings = load_settings(self.settings_path)
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await interaction.response.send_message(
                        f"Data file not found: {data_file} (use `!data list`)",
                        ephemeral=True,
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        var1, err = self._parse_grid_var(base_settings, tokens[0])
        if err:
            await interaction.response.send_message(
                f"Invalid variable: {err}",
                ephemeral=True,
            )
            return
        values1, err = self._build_grid_values(tokens[1], tokens[2], tokens[3])
        if err:
            await interaction.response.send_message(
                f"Invalid range: {err}",
                ephemeral=True,
            )
            return

        max_runs = 50
        total_runs = len(values1)
        if total_runs > max_runs:
            await interaction.response.send_message(
                f"Too many combinations ({total_runs}). Please keep it under {max_runs}.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        progress_msg = await interaction.followup.send(
            f"Running comp backtests (0/{total_runs})...",
            wait=True,
        )
        progress_channel = progress_msg.channel or interaction.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(interaction.channel.id)
        )
        progress_handle = self._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running comp backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Comp progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Comp progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running comp backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Comp progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self._run_comp_backtests,
                base_settings,
                var1,
                values1,
                progress_cb,
                total_runs,
            )
        finally:
            self._unregister_grid_progress(progress_handle)

        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        result_channel = await self._get_result_channel()
        target_channel = result_channel or interaction.channel or progress_channel
        if target_channel is None:
            await interaction.followup.send(
                "Comp backtest complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return

        for entry in results:
            value = entry.get("value")
            error = entry.get("error")
            if error:
                message = f"Comp failed for `{var1['label']}={value}`: {error}"
                await target_channel.send(content=message)
                continue
            stats = entry.get("stats", {})
            message = self._format_comp_output(
                var1,
                value,
                stats,
                data_label=str(data_pattern),
            )
            chart_path = entry.get("chart_path")
            chart_paths = [chart_path] if chart_path else None
            await self._send_message_and_charts(
                target_channel.send,
                message,
                chart_paths,
            )

        if result_channel is not None:
            await interaction.followup.send(
                f"Comp backtest complete. Results posted in #{result_channel.name}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Comp backtest complete. Results posted here.",
                ephemeral=True,
            )

    async def _run_costom3_from_interaction(
        self,
        interaction: discord.Interaction,
        tokens: List[str],
        data_file: Optional[str],
    ) -> None:
        if len(tokens) != 4:
            await interaction.response.send_message(
                "Usage: `!costom3 [--data FILE] section.key start end step`\n"
                "Example: `!costom3 --data BTC.csv strategy.params.fast_window 5 20 1`",
                ephemeral=True,
            )
            return
        base_settings = load_settings(self.settings_path)
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await interaction.response.send_message(
                        f"Data file not found: {data_file} (use `!data list`)",
                        ephemeral=True,
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        var1, err = self._parse_grid_var(base_settings, tokens[0])
        if err:
            await interaction.response.send_message(
                f"Invalid variable: {err}",
                ephemeral=True,
            )
            return
        values1, err = self._build_grid_values(tokens[1], tokens[2], tokens[3])
        if err:
            await interaction.response.send_message(
                f"Invalid range: {err}",
                ephemeral=True,
            )
            return

        max_runs = 50
        total_runs = len(values1)
        if total_runs > max_runs:
            await interaction.response.send_message(
                f"Too many combinations ({total_runs}). Please keep it under {max_runs}.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        progress_msg = await interaction.followup.send(
            f"Running costom3 backtests (0/{total_runs})...",
            wait=True,
        )
        progress_channel = progress_msg.channel or interaction.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(interaction.channel.id)
        )
        progress_handle = self._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running costom3 backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Costom3 progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Costom3 progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running costom3 backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Costom3 progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self._run_costom3_backtests,
                base_settings,
                var1,
                values1,
                progress_cb,
                total_runs,
            )
        finally:
            self._unregister_grid_progress(progress_handle)

        success_entries = [entry for entry in results if entry.get("result")]
        failed_values = [entry.get("value") for entry in results if entry.get("error")]
        if not success_entries:
            await interaction.followup.send(
                "Costom3 failed for all runs.",
                ephemeral=True,
            )
            return

        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        with tempfile.NamedTemporaryFile(
            prefix="costom3_overlay_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            output_path = Path(tmp_file.name)

        series = [
            {"label": str(entry.get("value")), "value": entry.get("value"), "result": entry.get("result")}
            for entry in success_entries
        ]
        chart_path = await asyncio.to_thread(
            create_equity_overlay_monthly_best_chart,
            series,
            output_path,
            title=f"Costom3 overlay ({var1['label']}) - {data_pattern}",
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or interaction.channel or progress_channel
        if target_channel is None:
            await interaction.followup.send(
                "Costom3 complete, but I couldn't find a channel to post results.",
                ephemeral=True,
            )
            return

        failed_note = ""
        if failed_values:
            preview = ", ".join(str(val) for val in failed_values[:6])
            suffix = "..." if len(failed_values) > 6 else ""
            failed_note = f"\n⚠️  Failed: {preview}{suffix}"
        message = (
            "**📈 Costom3 overlay**\n"
            f"🗂️  Data: `{data_pattern}`\n"
            f"🧩 Var: `{var1['label']}` | Range: `{tokens[1]} → {tokens[2]} (step {tokens[3]})`\n"
            f"✅ Runs: {len(success_entries)}/{len(results)}{failed_note}"
        )
        chart_paths = [chart_path] if chart_path else None
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )

        if result_channel is not None:
            await interaction.followup.send(
                f"Costom3 complete. Results posted in #{result_channel.name}.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Costom3 complete. Results posted here.",
                ephemeral=True,
            )

    def _flatten_section(self, section: str, data: Dict[str, Any]) -> List[tuple[str, Any]]:
        results: List[tuple[str, Any]] = []
        for key, value in data.items():
            if isinstance(value, dict):
                nested = self._flatten_section(f"{section}.{key}", value)
                results.extend(nested)
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                results.append((f"{section}.{key}", value))
                continue
            if isinstance(value, list) and all(
                isinstance(item, (str, int, float, bool)) for item in value
            ):
                results.append((f"{section}.{key}", value))
                for idx, item in enumerate(value, start=1):
                    results.append((f"{section}.{key}[{idx}]", item))
        return results

    async def _send_backtest_result(
        self,
        message: str,
        interaction: discord.Interaction,
        chart_paths: Optional[List[Path]] = None,
        view: Optional[View] = None,
    ) -> None:
        result_channel = await self._get_result_channel()
        if result_channel is None:
            await self._send_message_and_charts(
                interaction.followup.send,
                message,
                chart_paths,
                view=view,
            )
            return
        await self._send_message_and_charts(
            result_channel.send,
            message,
            chart_paths,
            view=view,
        )
        try:
            notice = f"Backtest complete. Results posted in #{result_channel.name}."
            if interaction.response.is_done():
                await interaction.followup.send(notice, ephemeral=True)
            else:
                await interaction.response.send_message(notice, ephemeral=True)
        except Exception:
            pass

    async def _send_rerun_prompt(
        self,
        interaction: discord.Interaction,
        view: View,
    ) -> None:
        result_channel = await self._get_result_channel()
        if result_channel:
            content = (
                f"Backtest complete. Results posted in #{result_channel.name}."
                " Run again?"
            )
        else:
            content = "Backtest complete. Run again?"
        if interaction.response.is_done():
            await interaction.followup.send(content=content, view=view)
        else:
            await interaction.response.send_message(content=content, view=view)

    async def _send_message_and_charts(
        self,
        send_func,
        message: str,
        chart_paths: Optional[List[Path]] = None,
        view: Optional[View] = None,
    ) -> None:
        chunks = self._split_message(message)
        for idx, chunk in enumerate(chunks):
            if chunk or view:
                await send_func(content=chunk, view=view if idx == 0 else None)
        if not chart_paths:
            return
        for path in chart_paths:
            if not path:
                continue
            await send_func(content="", files=[discord.File(path)])

    async def _send_embed_and_charts(
        self,
        send_func,
        embed: discord.Embed,
        chart_paths: Optional[List[Path]] = None,
    ) -> None:
        await send_func(embed=embed)
        if not chart_paths:
            return
        for path in chart_paths:
            if not path:
                continue
            await send_func(content="", files=[discord.File(path)])

    async def _execute_backtest(
        self,
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
    ) -> tuple[str, Optional[List[Path]]]:
        self.settings = load_settings(self.settings_path)
        loop = asyncio.get_running_loop()
        message, _, chart_paths, result, data = await loop.run_in_executor(
            None,
            run_backtest_with_result,
            self.settings,
            data_overrides,
            backtest_overrides,
            strategy_overrides,
        )
        if chart_paths is None:
            chart_paths = []
        has_settings_snapshot = any(
            path and path.suffix.lower() in {".yaml", ".yml"} and path.name.startswith("backtest_settings_")
            for path in chart_paths
        )
        if not has_settings_snapshot:
            snapshot = create_settings_snapshot(
                self.settings,
                data_overrides=data_overrides,
                backtest_overrides=backtest_overrides,
                strategy_overrides=strategy_overrides,
            )
            chart_paths.append(snapshot)
        self._last_backtest = {"result": result, "data": data}
        return message, chart_paths

    async def _execute_entry_diff(
        self,
        data_overrides: Dict[str, Any],
        backtest_overrides: Dict[str, Any],
        strategy_overrides: Dict[str, Any],
        diff_config: EntryDiffConfig,
    ) -> tuple[str, Optional[List[Path]]]:
        self.settings = load_settings(self.settings_path)
        loop = asyncio.get_running_loop()
        message, _, chart_paths, result, data = await loop.run_in_executor(
            None,
            run_backtest_with_entry_diff_result,
            self.settings,
            data_overrides,
            backtest_overrides,
            strategy_overrides,
            None,
            diff_config,
        )
        self._last_backtest = {"result": result, "data": data}
        return message, chart_paths

    def _select_chart_trades(
        self,
        mode: str,
        count: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if self._last_backtest is None:
            return [], 0
        result = self._last_backtest.get("result")
        if result is None:
            return [], 0
        exit_trades = [trade for trade in result.trades if trade.get("type") == "exit"]
        if mode == "profit":
            candidates = [trade for trade in exit_trades if float(trade.get("return", 0.0)) > 0.0]
            candidates.sort(key=lambda trade: float(trade.get("return", 0.0)), reverse=True)
        else:
            candidates = [trade for trade in exit_trades if float(trade.get("return", 0.0)) < 0.0]
            candidates.sort(key=lambda trade: float(trade.get("return", 0.0)))
        return candidates[:count], len(candidates)

    def _build_trade_chart_payloads(
        self,
        selected: List[Dict[str, Any]],
    ) -> List[Tuple[str, Path]]:
        if self._last_backtest is None:
            return []
        result = self._last_backtest.get("result")
        data = self._last_backtest.get("data")
        if result is None or data is None:
            return []
        if not selected:
            return []

        payloads: List[Tuple[str, Path]] = []
        total = len(selected)
        for idx, trade in enumerate(selected, start=1):
            ret_pct = float(trade.get("return", 0.0)) * 100.0
            reason = str(trade.get("entry_reason", "")).strip() or "N/A"
            direction = "Long" if trade.get("direction") == 1 else "Short"
            title = f"{direction} trade {idx}/{total} | Return {ret_pct:+.2f}%"
            with tempfile.NamedTemporaryFile(
                prefix="backtest_trade_",
                suffix=".png",
                delete=False,
            ) as tmp_file:
                chart_path = create_trade_snapshot_chart(
                    data,
                    trade,
                    Path(tmp_file.name),
                    title=title,
                )
            if chart_path:
                message = f"🔎 Entry reason: {reason}"
                payloads.append((message, chart_path))
        return payloads

    async def _send_trade_charts(
        self,
        interaction: discord.Interaction,
        mode: str,
        count: int,
    ) -> None:
        if self._last_backtest is None:
            await interaction.followup.send(
                "Run `!testrun` first so I have results to chart.",
                ephemeral=True,
            )
            return

        selected, available = self._select_chart_trades(mode, count)
        if not selected:
            await interaction.followup.send(
                "No matching trades found for that selection.",
                ephemeral=True,
            )
            return

        loop = asyncio.get_running_loop()
        payloads = await loop.run_in_executor(
            None,
            self._build_trade_chart_payloads,
            selected,
        )
        if not payloads:
            await interaction.followup.send(
                "No matching trades found for that selection.",
                ephemeral=True,
            )
            return

        result_channel = await self._get_result_channel()
        destination = result_channel.send if result_channel else interaction.followup.send

        selection = "profit" if mode == "profit" else "loss"
        header = f"Trade charts ({selection}, showing {len(selected)}/{available})"
        await destination(content=header)
        for message, chart_path in payloads:
            await destination(content=message, files=[discord.File(chart_path)])

        if result_channel:
            try:
                notice = f"Charts posted in #{result_channel.name}."
                await interaction.followup.send(notice, ephemeral=True)
            except Exception:
                pass

    @app_commands.command(name="testrun", description="Run a backtest on selected data.")
    @app_commands.describe(
        strategy="Strategy name",
        data_file="Price data CSV filename",
        initial_balance="Initial balance",
        fee_rate="Fee rate per trade",
        leverage="Leverage",
        position_size="Position size (fraction)",
        slippage="Slippage rate",
        fast_window="Fast window for SMA",
        slow_window="Slow window for SMA",
    )
    async def backtest_slash(
        self,
        interaction: discord.Interaction,
        strategy: Optional[str] = None,
        data_file: Optional[str] = None,
        initial_balance: Optional[float] = None,
        fee_rate: Optional[float] = None,
        leverage: Optional[float] = None,
        position_size: Optional[float] = None,
        slippage: Optional[float] = None,
        fast_window: Optional[int] = None,
        slow_window: Optional[int] = None,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can run backtests.", ephemeral=True
            )
            return
        if not self._is_backtest_channel(interaction.channel):
            await interaction.response.send_message(
                self._backtest_channel_notice(),
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        data_overrides: Dict[str, Any] = {}
        if data_file:
            data_overrides["file_pattern"] = data_file

        backtest_overrides: Dict[str, Any] = {}
        if initial_balance is not None:
            backtest_overrides["initial_balance"] = initial_balance
        if fee_rate is not None:
            backtest_overrides["fee_rate"] = fee_rate
        if leverage is not None:
            backtest_overrides["leverage"] = leverage
        if position_size is not None:
            backtest_overrides["position_size"] = position_size
        if slippage is not None:
            backtest_overrides["slippage"] = slippage

        strategy_overrides: Dict[str, Any] = {}
        if strategy:
            strategy_overrides["name"] = strategy
        params: Dict[str, Any] = {}
        if fast_window is not None:
            params["fast_window"] = fast_window
        if slow_window is not None:
            params["slow_window"] = slow_window
        if params:
            strategy_overrides["params"] = params

        strategy_overrides, err = self._enforce_cheatkey_strategy(
            interaction.channel,
            strategy_overrides,
        )
        if err:
            await interaction.followup.send(err, ephemeral=True)
            return

        try:
            message, chart_paths = await self._execute_backtest(
                data_overrides,
                backtest_overrides,
                strategy_overrides,
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Backtest failed: {exc}", ephemeral=True
            )
            return
        await self._send_backtest_result(message, interaction, chart_paths)

    async def backtest_data_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        return await self._data_file_autocomplete(interaction, current)

    async def backtest_strategy_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        return await self._strategy_autocomplete(interaction, current)

    async def download_interval_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> List[app_commands.Choice[str]]:
        return await self._interval_autocomplete(interaction, current)

    async def _handle_download_request(
        self,
        interaction: discord.Interaction,
        symbol: str,
        interval: str,
        start: str,
        end: str,
        market: str,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can download data.", ephemeral=True
            )
            return
        if not self._is_command_channel(interaction.channel):
            await interaction.response.send_message(
                f"Please use #{self.command_channel_name} for data downloads.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        output_dir = self.settings.data.get("folder", "price data")
        loop = asyncio.get_running_loop()
        try:
            file_path = await loop.run_in_executor(
                None,
                download_price_data,
                symbol,
                interval,
                start,
                end,
                output_dir,
                (market or "futures").lower(),
            )
        except Exception as exc:
            await interaction.followup.send(
                f"Download failed: {exc}", ephemeral=True
            )
            return

        self._set_download_defaults(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "market": market or "futures",
            }
        )

        await interaction.followup.send(
            f"Downloaded data to {file_path.name}.", ephemeral=True
        )
        await self._update_data_list_channel()
        await self._update_strategy_list_channel()

    @app_commands.command(
        name="download",
        description="Download price data (opens a form).",
    )
    async def download_data_slash(
        self,
        interaction: discord.Interaction,
    ) -> None:
        await interaction.response.send_modal(DownloadDataModal(self))

    @app_commands.command(
        name="sync",
        description="Sync slash commands for this server.",
    )
    async def sync_slash(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can sync commands.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        await self._sync_commands(guild=interaction.guild)
        await interaction.followup.send(
            "Synced commands for this server.", ephemeral=True
        )

    async def sync_text(self, ctx: commands.Context) -> None:
        if not self.enable_slash:
            await ctx.reply("Slash commands are disabled; using ! commands only.")
            return
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can sync commands.")
            return
        await ctx.reply("Syncing commands for this server...")
        await self._sync_commands(guild=ctx.guild)
        await ctx.reply("Synced commands for this server.")

    async def download_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can download data.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(
                f"Please use #{self.command_channel_name} for data downloads."
            )
            return
        view = DownloadDataButtonView(self)
        await ctx.reply("Click the button to open the download form.", view=view)

    async def _delete_data_file(self, filename: str) -> tuple[bool, str]:
        path = self._resolve_price_file(filename)
        if path is None:
            return False, "File not found in the data folder."
        try:
            path.unlink()
        except Exception as exc:
            return False, f"Failed to delete {filename}: {exc}"
        return True, f"Deleted {filename}."

    async def delete_data_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can delete data.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(
                f"Please use #{self.command_channel_name} to delete data."
            )
            return

        filename = args.strip()
        if filename:
            deleted, message = await self._delete_data_file(filename)
            await ctx.reply(message)
            if deleted:
                await self._update_data_list_channel()
                await self._update_strategy_list_channel()
            return

        files = self._list_price_files()
        if not files:
            await ctx.reply("No data files to delete.")
            return
        if len(files) > 25:
            await ctx.reply(
                "Too many files for the picker. "
                "Use `!delete_data <filename>` or delete from the latest 25 files."
            )
            files = files[-25:]
        view = DeleteDataView(self, files)
        await ctx.reply("Select a data file to delete:", view=view)

    async def backtest_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        parsed = self._parse_kv_tokens(tokens)
        if not parsed and args.strip():
            await ctx.reply(
                "Usage: `!testrun key=value ...` "
                "(keys: strategy, data_file, initial_balance, fee_rate, leverage, "
                "position_size, slippage, fast_window, slow_window)"
            )
            return

        data_overrides: Dict[str, Any] = {}
        backtest_overrides: Dict[str, Any] = {}
        strategy_overrides: Dict[str, Any] = {}
        params: Dict[str, Any] = {}

        for key, value in parsed.items():
            try:
                coerced = self._coerce_backtest_arg(key, value)
            except ValueError:
                await ctx.reply(f"Invalid value for {key}: {value}")
                return
            if key == "data_file":
                data_overrides["file_pattern"] = coerced
            elif key == "strategy":
                strategy_overrides["name"] = coerced
            elif key in {"initial_balance", "fee_rate", "leverage", "position_size", "slippage"}:
                backtest_overrides[key] = coerced
            elif key in {"fast_window", "slow_window"}:
                params[key] = coerced
            else:
                await ctx.reply(f"Unknown key: {key}")
                return

        if params:
            strategy_overrides["params"] = params

        strategy_overrides, err = self._enforce_cheatkey_strategy(
            ctx.channel,
            strategy_overrides,
        )
        if err:
            await ctx.reply(err)
            return

        if "file_pattern" not in data_overrides:
            files = self._list_price_files()
            if not files:
                await ctx.reply("No data files found. Download data first.")
                return
            notice = "Select a data file to run the backtest:"
            if len(files) > 25:
                notice = (
                    "Select a data file to run the backtest "
                    "(showing latest 25 files):"
                )
                files = files[-25:]
            view = BacktestDataView(
                self,
                files,
                data_overrides,
                backtest_overrides,
                strategy_overrides,
                ctx.author.id,
            )
            await ctx.reply(notice, view=view)
            return

        await ctx.reply("Running backtest...")
        try:
            message, chart_paths = await self._execute_backtest(
                data_overrides,
                backtest_overrides,
                strategy_overrides,
            )
        except Exception as exc:
            await ctx.reply(f"Backtest failed: {exc}")
            return

        result_channel = await self._get_result_channel()
        if result_channel is None:
            await self._send_message_and_charts(
                ctx.reply,
                message,
                chart_paths,
            )
            await ctx.reply(
                "Backtest complete. Run again?",
                view=RerunBacktestView(
                    self,
                    ctx.author.id,
                    data_overrides,
                    backtest_overrides,
                    strategy_overrides,
                ),
            )
            return
        await self._send_message_and_charts(
            result_channel.send,
            message,
            chart_paths,
        )
        await ctx.reply(
            f"Backtest complete. Results posted in #{result_channel.name}. Run again?",
            view=RerunBacktestView(
                self,
                ctx.author.id,
                data_overrides,
                backtest_overrides,
                strategy_overrides,
            ),
        )

    async def entry_diff_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run this analysis.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        parsed = self._parse_kv_tokens(tokens)
        if not parsed and args.strip():
            await ctx.reply(
                "Usage: `!diff key=value ...` "
                "(keys: strategy, data_file, initial_balance, fee_rate, leverage, "
                "position_size, slippage, fast_window, slow_window, "
                "min_samples, top_features, max_groups)"
            )
            return

        data_overrides: Dict[str, Any] = {}
        backtest_overrides: Dict[str, Any] = {}
        strategy_overrides: Dict[str, Any] = {}
        params: Dict[str, Any] = {}
        diff_config = EntryDiffConfig()

        for key, value in parsed.items():
            if key == "data_file":
                data_overrides["file_pattern"] = value
                continue
            if key == "strategy":
                strategy_overrides["name"] = value
                continue
            if key in {"min_samples", "top_features", "max_groups"}:
                try:
                    coerced = int(value)
                except ValueError:
                    await ctx.reply(f"Invalid value for {key}: {value}")
                    return
                if key == "min_samples":
                    diff_config.min_side_samples = coerced
                elif key == "top_features":
                    diff_config.top_features = coerced
                else:
                    diff_config.max_groups = coerced
                continue
            try:
                coerced = self._coerce_backtest_arg(key, value)
            except ValueError:
                await ctx.reply(f"Invalid value for {key}: {value}")
                return
            if key in {"initial_balance", "fee_rate", "leverage", "position_size", "slippage"}:
                backtest_overrides[key] = coerced
            elif key in {"fast_window", "slow_window"}:
                params[key] = coerced
            else:
                await ctx.reply(f"Unknown key: {key}")
                return

        if params:
            strategy_overrides["params"] = params

        strategy_overrides, err = self._enforce_cheatkey_strategy(
            ctx.channel,
            strategy_overrides,
        )
        if err:
            await ctx.reply(err)
            return

        if "file_pattern" not in data_overrides:
            files = self._list_price_files()
            if not files:
                await ctx.reply("No data files found. Download data first.")
                return
            notice = "Select a data file to run entry diff:"
            if len(files) > 25:
                notice = (
                    "Select a data file to run entry diff "
                    "(showing latest 25 files):"
                )
                files = files[-25:]
            view = EntryDiffDataView(
                self,
                files,
                data_overrides,
                backtest_overrides,
                strategy_overrides,
                diff_config,
                ctx.author.id,
            )
            await ctx.reply(notice, view=view)
            return

        await ctx.reply("Running entry diff...")
        try:
            message, chart_paths = await self._execute_entry_diff(
                data_overrides,
                backtest_overrides,
                strategy_overrides,
                diff_config,
            )
        except Exception as exc:
            await ctx.reply(f"Entry diff failed: {exc}")
            return

        result_channel = await self._get_result_channel()
        if result_channel is None:
            await self._send_message_and_charts(
                ctx.reply,
                message,
                chart_paths,
            )
            return
        await self._send_message_and_charts(
            result_channel.send,
            message,
            chart_paths,
        )
        await ctx.reply(
            f"Entry diff complete. Results posted in #{result_channel.name}."
        )

    async def grid_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        data_file: Optional[str] = None
        if tokens:
            filtered: List[str] = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--data":
                    if idx + 1 >= len(tokens):
                        await ctx.reply("Missing value after `--data`.")
                        return
                    data_file = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("--data="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                if token.startswith("data=") or token.startswith("data_file=") or token.startswith("file="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                filtered.append(token)
                idx += 1
            tokens = filtered
        if not tokens:
            view = GridSetupView(self, ctx.author.id)
            message = await ctx.reply(view._summary(), view=view)
            view.message = message
            return
        if len(tokens) not in {4, 8}:
            await ctx.reply(
                "Usage: `!grid [--data FILE] section.key start end step "
                "[section.key start end step]`\n"
                "Example: `!grid --data BTC.csv strategy.params.fast_window 5 10 1 "
                "strategy.params.slow_window 5 10 1`"
            )
            return
        if data_file is None:
            files = self._list_price_files()
            if files:
                if len(files) > 25:
                    notice = (
                        "Select a data file for grid "
                        "(showing latest 25 files, or Default to use settings):"
                    )
                    files = files[-25:]
                else:
                    notice = "Select a data file for grid (or Default to use settings):"
                view = GridDataFileView(self, files, tokens, ctx.author.id)
                await ctx.reply(notice, view=view)
                return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, ctx.channel)
        if err:
            await ctx.reply(err)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await ctx.reply(
                        f"Data file not found: {data_file} (use `!data list`)"
                    )
                    return
            base_settings.data["file_pattern"] = data_file
        var1, err = self._parse_grid_var(base_settings, tokens[0])
        if err:
            await ctx.reply(f"Invalid first variable: {err}")
            return
        values1, err = self._build_grid_values(tokens[1], tokens[2], tokens[3])
        if err:
            await ctx.reply(f"Invalid first range: {err}")
            return

        var2: Optional[dict] = None
        values2: Optional[List[Any]] = None
        if len(tokens) == 8:
            var2, err = self._parse_grid_var(base_settings, tokens[4])
            if err:
                await ctx.reply(f"Invalid second variable: {err}")
                return
            values2, err = self._build_grid_values(tokens[5], tokens[6], tokens[7])
            if err:
                await ctx.reply(f"Invalid second range: {err}")
                return

        max_runs = 250
        total_runs = len(values1) * (len(values2) if values2 else 1)
        if total_runs > max_runs:
            await ctx.reply(
                f"Too many combinations ({total_runs}). "
                f"Please keep it under {max_runs}."
            )
            return

        progress_msg = await ctx.reply(
            f"Running grid backtests (0/{total_runs})..."
        )
        progress_channel = progress_msg.channel or ctx.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(ctx.channel.id)
        )
        progress_handle = self._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running grid backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Grid progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Grid progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running grid backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Grid progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self._run_grid_backtests,
                base_settings,
                var1,
                values1,
                var2,
                values2,
                progress_cb,
                total_runs,
            )
        finally:
            self._unregister_grid_progress(progress_handle)
        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        output = self._format_grid_output(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )
        image_paths = self._build_grid_heatmap_images(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )
        data_paths = self._build_grid_data_file(
            var1,
            values1,
            var2,
            values2,
            results,
            data_label=str(data_pattern),
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or ctx.channel
        if target_channel is None:
            await ctx.reply(
                "Grid backtest complete, but I couldn't find a channel to post results."
            )
            return
        sent = await self._send_grid_output_to_channel(
            target_channel.id,
            output,
            image_paths=image_paths,
            data_paths=data_paths,
        )
        if not sent:
            self._queue_grid_output(
                target_channel.id,
                output,
                image_paths=image_paths,
                data_paths=data_paths,
            )
        if result_channel is not None:
            await ctx.reply(
                f"Grid backtest complete. Results posted in #{result_channel.name}."
            )
        else:
            await ctx.reply("Grid backtest complete. Results posted here.")

    async def comp_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        data_file: Optional[str] = None
        if tokens:
            filtered: List[str] = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--data":
                    if idx + 1 >= len(tokens):
                        await ctx.reply("Missing value after `--data`.")
                        return
                    data_file = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("--data="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                if token.startswith("data=") or token.startswith("data_file=") or token.startswith("file="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                filtered.append(token)
                idx += 1
            tokens = filtered

        if not tokens:
            view = CompSetupView(self, ctx.author.id)
            message = await ctx.reply(view._summary(), view=view)
            view.message = message
            return

        if len(tokens) != 4:
            await ctx.reply(
                "Usage: `!comp [--data FILE] section.key start end step`\n"
                "Example: `!comp --data BTC.csv strategy.params.fast_window 5 20 1`"
            )
            return
        if data_file is None:
            files = self._list_price_files()
            if files:
                if len(files) > 25:
                    notice = (
                        "Select a data file for comp "
                        "(showing latest 25 files, or Default to use settings):"
                    )
                    files = files[-25:]
                else:
                    notice = "Select a data file for comp (or Default to use settings):"
                view = CompDataFileView(self, files, tokens, ctx.author.id)
                await ctx.reply(notice, view=view)
                return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, ctx.channel)
        if err:
            await ctx.reply(err)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await ctx.reply(
                        f"Data file not found: {data_file} (use `!data list`)"
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        var1, err = self._parse_grid_var(base_settings, tokens[0])
        if err:
            await ctx.reply(f"Invalid variable: {err}")
            return
        values1, err = self._build_grid_values(tokens[1], tokens[2], tokens[3])
        if err:
            await ctx.reply(f"Invalid range: {err}")
            return

        max_runs = 50
        total_runs = len(values1)
        if total_runs > max_runs:
            await ctx.reply(
                f"Too many combinations ({total_runs}). "
                f"Please keep it under {max_runs}."
            )
            return

        progress_msg = await ctx.reply(
            f"Running comp backtests (0/{total_runs})..."
        )
        progress_channel = progress_msg.channel or ctx.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(ctx.channel.id)
        )
        progress_handle = self._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running comp backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Comp progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Comp progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running comp backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Comp progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self._run_comp_backtests,
                base_settings,
                var1,
                values1,
                progress_cb,
                total_runs,
            )
        finally:
            self._unregister_grid_progress(progress_handle)

        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        result_channel = await self._get_result_channel()
        target_channel = result_channel or ctx.channel
        if target_channel is None:
            await ctx.reply(
                "Comp backtest complete, but I couldn't find a channel to post results."
            )
            return

        for entry in results:
            value = entry.get("value")
            error = entry.get("error")
            if error:
                message = f"Comp failed for `{var1['label']}={value}`: {error}"
                await target_channel.send(content=message)
                continue
            stats = entry.get("stats", {})
            message = self._format_comp_output(
                var1,
                value,
                stats,
                data_label=str(data_pattern),
            )
            chart_path = entry.get("chart_path")
            chart_paths = [chart_path] if chart_path else None
            await self._send_message_and_charts(
                target_channel.send,
                message,
                chart_paths,
            )

        if result_channel is not None:
            await ctx.reply(
                f"Comp backtest complete. Results posted in #{result_channel.name}."
            )
        else:
            await ctx.reply("Comp backtest complete. Results posted here.")

    async def costom_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        data_file: Optional[str] = None
        if tokens:
            filtered: List[str] = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--data":
                    if idx + 1 >= len(tokens):
                        await ctx.reply("Missing value after `--data`.")
                        return
                    data_file = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("--data="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                if token.startswith("data=") or token.startswith("data_file=") or token.startswith("file="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                filtered.append(token)
                idx += 1
            tokens = filtered

        if not tokens:
            view = CustomSetupView(self, ctx.author.id)
            message = await ctx.reply(view._summary(), view=view)
            view.message = message
            return

        if len(tokens) != 2:
            await ctx.reply(
                "Usage: `!costom [--data FILE] lookback buffer_percent`\n"
                "Example: `!costom --data BTC.csv 20 0.1`"
            )
            return

        if data_file is None:
            files = self._list_price_files()
            if files:
                if len(files) > 25:
                    notice = (
                        "Select a data file for costom "
                        "(showing latest 25 files, or Default to use settings):"
                    )
                    files = files[-25:]
                else:
                    notice = "Select a data file for costom (or Default to use settings):"
                view = CustomDataFileView(self, files, tokens, ctx.author.id)
                await ctx.reply(notice, view=view)
                return

        lookback, buffer_pct, err = self._parse_costom_params(tokens[0], tokens[1])
        if err:
            await ctx.reply(err)
            return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, ctx.channel)
        if err:
            await ctx.reply(err)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await ctx.reply(
                        f"Data file not found: {data_file} (use `!data list`)"
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        await ctx.reply("Running costom...")
        loop = asyncio.get_running_loop()
        settings_copy = copy.deepcopy(base_settings)
        chart_cfg = settings_copy.backtest.get("chart")
        if isinstance(chart_cfg, dict):
            chart_cfg = dict(chart_cfg)
            chart_cfg["enabled"] = False
        else:
            chart_cfg = {"enabled": False}
        settings_copy.backtest["chart"] = chart_cfg

        try:
            _, _, _, result, data = await loop.run_in_executor(
                None,
                run_backtest_with_result,
                settings_copy,
            )
        except Exception as exc:
            await ctx.reply(f"Costom failed: {exc}")
            return

        self._last_backtest = {"result": result, "data": data}
        data_pattern = settings_copy.data.get("file_pattern", "*.csv")
        message, chart_paths, _, _, _ = await asyncio.to_thread(
            BacktestBot._compute_costom_bundle,
            result,
            data,
            lookback,
            buffer_pct,
            settings_copy,
            str(data_pattern),
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or ctx.channel
        if target_channel is None:
            await ctx.reply(
                "Costom complete, but I couldn't find a channel to post results."
            )
            return
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )
        if result_channel is not None:
            await ctx.reply(
                f"Costom complete. Results posted in #{result_channel.name}."
            )
        else:
            await ctx.reply("Costom complete. Results posted here.")

    async def costom2_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        data_file: Optional[str] = None
        if tokens:
            filtered: List[str] = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--data":
                    if idx + 1 >= len(tokens):
                        await ctx.reply("Missing value after `--data`.")
                        return
                    data_file = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("--data="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                if token.startswith("data=") or token.startswith("data_file=") or token.startswith("file="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                filtered.append(token)
                idx += 1
            tokens = filtered

        if tokens:
            await ctx.reply("Usage: `!costom2 [--data FILE]`")
            return

        if data_file is None:
            files = self._list_price_files()
            if files:
                if len(files) > 25:
                    notice = (
                        "Select a data file for costom2 "
                        "(showing latest 25 files, or Default to use settings):"
                    )
                    files = files[-25:]
                else:
                    notice = "Select a data file for costom2 (or Default to use settings):"
                view = Costom2DataFileView(self, files, tokens, ctx.author.id)
                await ctx.reply(notice, view=view)
                return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, ctx.channel)
        if err:
            await ctx.reply(err)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await ctx.reply(
                        f"Data file not found: {data_file} (use `!data list`)"
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        await ctx.reply("Running costom2...")
        loop = asyncio.get_running_loop()
        settings_copy = copy.deepcopy(base_settings)
        chart_cfg = settings_copy.backtest.get("chart")
        if isinstance(chart_cfg, dict):
            chart_cfg = dict(chart_cfg)
            chart_cfg["enabled"] = False
        else:
            chart_cfg = {"enabled": False}
        settings_copy.backtest["chart"] = chart_cfg

        try:
            _, _, _, result, _ = await loop.run_in_executor(
                None,
                run_backtest_with_result,
                settings_copy,
            )
        except Exception as exc:
            await ctx.reply(f"Costom2 failed: {exc}")
            return

        self._last_backtest = {"result": result, "data": None}
        data_pattern = settings_copy.data.get("file_pattern", "*.csv")
        with tempfile.NamedTemporaryFile(
            prefix="costom2_monthly_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            output_path = Path(tmp_file.name)
        chart_path = await asyncio.to_thread(
            create_equity_mdd_monthly_stats_chart,
            result,
            output_path,
            title=f"Equity + MDD + Monthly stats ({data_pattern})",
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or ctx.channel
        if target_channel is None:
            await ctx.reply(
                "Costom2 complete, but I couldn't find a channel to post results."
            )
            return

        message = f"**📊 Costom2 monthly stats**\n🗂️  Data: `{data_pattern}`"
        chart_paths = [chart_path] if chart_path else None
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )

        if result_channel is not None:
            await ctx.reply(
                f"Costom2 complete. Results posted in #{result_channel.name}."
            )
        else:
            await ctx.reply("Costom2 complete. Results posted here.")

    async def costom4_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        data_file: Optional[str] = None
        if tokens:
            filtered: List[str] = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--data":
                    if idx + 1 >= len(tokens):
                        await ctx.reply("Missing value after `--data`.")
                        return
                    data_file = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("--data="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                if token.startswith("data=") or token.startswith("data_file=") or token.startswith("file="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                filtered.append(token)
                idx += 1
            tokens = filtered

        if tokens:
            await ctx.reply("Usage: `!costom4 [--data FILE]`")
            return

        if data_file is None:
            files = self._list_price_files()
            if files:
                if len(files) > 25:
                    notice = (
                        "Select a data file for costom4 "
                        "(showing latest 25 files, or Default to use settings):"
                    )
                    files = files[-25:]
                else:
                    notice = "Select a data file for costom4 (or Default to use settings):"
                view = Costom4DataFileView(self, files, tokens, ctx.author.id)
                await ctx.reply(notice, view=view)
                return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, ctx.channel)
        if err:
            await ctx.reply(err)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await ctx.reply(
                        f"Data file not found: {data_file} (use `!data list`)"
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        await ctx.reply("Running costom4...")
        loop = asyncio.get_running_loop()
        settings_copy = copy.deepcopy(base_settings)
        chart_cfg = settings_copy.backtest.get("chart")
        if isinstance(chart_cfg, dict):
            chart_cfg = dict(chart_cfg)
            chart_cfg["enabled"] = False
        else:
            chart_cfg = {"enabled": False}
        settings_copy.backtest["chart"] = chart_cfg

        try:
            _, _, _, result, data = await loop.run_in_executor(
                None,
                run_backtest_with_result,
                settings_copy,
            )
        except Exception as exc:
            await ctx.reply(f"Costom4 failed: {exc}")
            return

        self._last_backtest = {"result": result, "data": data}
        data_pattern = settings_copy.data.get("file_pattern", "*.csv")
        monthly_df, stats = compute_monthly_return_stats(result)
        entry_minute_df, entry_minute_stats = compute_entry_minute_win_rate(result)
        entry_minute_tp_df, entry_minute_tp_stats = compute_entry_minute_tp_rate(result)
        message = BacktestBot._format_costom4_output(
            data_label=str(data_pattern),
            monthly_df=monthly_df,
            stats=stats,
            entry_minute_df=entry_minute_df,
            entry_minute_stats=entry_minute_stats,
            entry_minute_tp_df=entry_minute_tp_df,
            entry_minute_tp_stats=entry_minute_tp_stats,
        )

        with tempfile.NamedTemporaryFile(
            prefix="costom4_monthly_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            output_path = Path(tmp_file.name)
        chart_path = await asyncio.to_thread(
            create_monthly_return_stability_chart,
            result,
            output_path,
            title=f"Monthly return stability ({data_pattern})",
        )
        with tempfile.NamedTemporaryFile(
            prefix="costom4_entry_minute_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            minute_output_path = Path(tmp_file.name)
        entry_minute_chart = await asyncio.to_thread(
            create_entry_minute_win_rate_chart,
            result,
            minute_output_path,
            title=f"Entry minute win rate ({data_pattern})",
        )
        with tempfile.NamedTemporaryFile(
            prefix="costom4_entry_minute_tp_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            tp_output_path = Path(tmp_file.name)
        entry_minute_tp_chart = await asyncio.to_thread(
            create_entry_minute_tp_rate_chart,
            result,
            tp_output_path,
            title=f"Entry minute TP reach rate ({data_pattern})",
        )
        equity_price_chart = None
        if isinstance(data, pd.DataFrame) and not data.empty:
            with tempfile.NamedTemporaryFile(
                prefix="costom4_equity_price_",
                suffix=".png",
                delete=False,
            ) as tmp_file:
                price_output_path = Path(tmp_file.name)
            equity_price_chart = await asyncio.to_thread(
                create_equity_price_chart,
                result,
                data,
                price_output_path,
            )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or ctx.channel
        if target_channel is None:
            await ctx.reply(
                "Costom4 complete, but I couldn't find a channel to post results."
            )
            return

        chart_paths = []
        if chart_path:
            chart_paths.append(chart_path)
        if entry_minute_chart:
            chart_paths.append(entry_minute_chart)
        if entry_minute_tp_chart:
            chart_paths.append(entry_minute_tp_chart)
        if equity_price_chart:
            chart_paths.append(equity_price_chart)
        if not chart_paths:
            chart_paths = None
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )

        if result_channel is not None:
            await ctx.reply(
                f"Costom4 complete. Results posted in #{result_channel.name}."
            )
        else:
            await ctx.reply("Costom4 complete. Results posted here.")

    async def costom3_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can run backtests.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        data_file: Optional[str] = None
        if tokens:
            filtered: List[str] = []
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token == "--data":
                    if idx + 1 >= len(tokens):
                        await ctx.reply("Missing value after `--data`.")
                        return
                    data_file = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("--data="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                if token.startswith("data=") or token.startswith("data_file=") or token.startswith("file="):
                    data_file = token.split("=", 1)[1]
                    idx += 1
                    continue
                filtered.append(token)
                idx += 1
            tokens = filtered

        if not tokens:
            view = Costom3SetupView(self, ctx.author.id)
            message = await ctx.reply(view._summary(), view=view)
            view.message = message
            return

        if len(tokens) != 4:
            await ctx.reply(
                "Usage: `!costom3 [--data FILE] section.key start end step`\n"
                "Example: `!costom3 --data BTC.csv strategy.params.fast_window 5 20 1`"
            )
            return
        if data_file is None:
            files = self._list_price_files()
            if files:
                if len(files) > 25:
                    notice = (
                        "Select a data file for costom3 "
                        "(showing latest 25 files, or Default to use settings):"
                    )
                    files = files[-25:]
                else:
                    notice = "Select a data file for costom3 (or Default to use settings):"
                view = Costom3DataFileView(self, files, tokens, ctx.author.id)
                await ctx.reply(notice, view=view)
                return

        base_settings = load_settings(self.settings_path)
        err = self._apply_forced_strategy(base_settings, ctx.channel)
        if err:
            await ctx.reply(err)
            return
        if data_file:
            if not any(ch in data_file for ch in "*?[]"):
                if data_file not in self._list_price_files():
                    await ctx.reply(
                        f"Data file not found: {data_file} (use `!data list`)"
                    )
                    return
            base_settings.data["file_pattern"] = data_file

        var1, err = self._parse_grid_var(base_settings, tokens[0])
        if err:
            await ctx.reply(f"Invalid variable: {err}")
            return
        values1, err = self._build_grid_values(tokens[1], tokens[2], tokens[3])
        if err:
            await ctx.reply(f"Invalid range: {err}")
            return

        max_runs = 50
        total_runs = len(values1)
        if total_runs > max_runs:
            await ctx.reply(
                f"Too many combinations ({total_runs}). "
                f"Please keep it under {max_runs}."
            )
            return

        progress_msg = await ctx.reply(
            f"Running costom3 backtests (0/{total_runs})..."
        )
        progress_channel = progress_msg.channel or ctx.channel
        channel_id = (
            int(progress_channel.id)
            if progress_channel is not None
            else int(ctx.channel.id)
        )
        progress_handle = self._register_grid_progress(
            progress_msg,
            channel_id,
            total_runs,
        )
        loop = asyncio.get_running_loop()
        progress_state = {"last_done": 0, "last_ts": 0.0}

        async def update_progress(done: int, total: int) -> None:
            progress_handle.done = done
            progress_handle.total = total
            pct = int((done / total) * 100) if total else 100
            message = progress_handle.message
            if message is not None:
                try:
                    await message.edit(
                        content=f"Running costom3 backtests ({done}/{total})... {pct}%"
                    )
                    return
                except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                    logger.warning(
                        "Costom3 progress update failed; posting new message: %s", exc
                    )
            channel = progress_channel or self.get_channel(progress_handle.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(progress_handle.channel_id)
                except Exception as exc:
                    logger.warning("Costom3 progress send failed: %s", exc)
                    return
            try:
                progress_handle.message = await channel.send(
                    f"Running costom3 backtests ({done}/{total})... {pct}%"
                )
            except discord.HTTPException as exc:
                logger.warning("Costom3 progress send failed: %s", exc)
                return

        def progress_cb(done: int, total: int) -> None:
            now = time.monotonic()
            step = max(1, total // 20) if total else 1
            if done < total:
                if done - progress_state["last_done"] < step and (now - progress_state["last_ts"]) < 2.0:
                    return
            progress_state["last_done"] = done
            progress_state["last_ts"] = now
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(update_progress(done, total))
            )

        try:
            results = await loop.run_in_executor(
                None,
                self._run_costom3_backtests,
                base_settings,
                var1,
                values1,
                progress_cb,
                total_runs,
            )
        finally:
            self._unregister_grid_progress(progress_handle)

        success_entries = [entry for entry in results if entry.get("result")]
        failed_values = [entry.get("value") for entry in results if entry.get("error")]
        if not success_entries:
            await ctx.reply("Costom3 failed for all runs.")
            return

        data_pattern = base_settings.data.get("file_pattern", "*.csv")
        with tempfile.NamedTemporaryFile(
            prefix="costom3_overlay_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            output_path = Path(tmp_file.name)

        series = [
            {"label": str(entry.get("value")), "value": entry.get("value"), "result": entry.get("result")}
            for entry in success_entries
        ]
        chart_path = await asyncio.to_thread(
            create_equity_overlay_monthly_best_chart,
            series,
            output_path,
            title=f"Costom3 overlay ({var1['label']}) - {data_pattern}",
        )

        result_channel = await self._get_result_channel()
        target_channel = result_channel or ctx.channel
        if target_channel is None:
            await ctx.reply(
                "Costom3 complete, but I couldn't find a channel to post results."
            )
            return

        failed_note = ""
        if failed_values:
            preview = ", ".join(str(val) for val in failed_values[:6])
            suffix = "..." if len(failed_values) > 6 else ""
            failed_note = f"\n⚠️  Failed: {preview}{suffix}"
        message = (
            "**📈 Costom3 overlay**\n"
            f"🗂️  Data: `{data_pattern}`\n"
            f"🧩 Var: `{var1['label']}` | Range: `{tokens[1]} → {tokens[2]} (step {tokens[3]})`\n"
            f"✅ Runs: {len(success_entries)}/{len(results)}{failed_note}"
        )
        chart_paths = [chart_path] if chart_path else None
        await self._send_message_and_charts(
            target_channel.send,
            message,
            chart_paths,
        )

        if result_channel is not None:
            await ctx.reply(
                f"Costom3 complete. Results posted in #{result_channel.name}."
            )
        else:
            await ctx.reply("Costom3 complete. Results posted here.")

    async def chart_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can request charts.")
            return
        if not self._is_backtest_channel(ctx.channel):
            await ctx.reply(self._backtest_channel_notice())
            return
        if args.strip():
            await ctx.reply("Usage: `!chart`")
            return
        if self._last_backtest is None:
            await ctx.reply("Run `!testrun` first so I have results to chart.")
            return

        view = ChartTypeView(self, ctx.author.id)
        await ctx.reply("Select profit or loss trades to chart:", view=view)

    async def set_config_text(
        self,
        ctx: commands.Context,
        section: Optional[str] = None,
        key: Optional[str] = None,
        *,
        value: Optional[str] = None,
    ) -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can change settings.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(
                f"Please use #{self.command_channel_name} for config changes."
            )
            return
        if not section or not key or value is None:
            await ctx.reply("Usage: `!set_config section key value`")
            return
        if section not in {"data", "backtest", "strategy"}:
            await ctx.reply("Section must be one of: data, backtest, strategy.")
            return

        coerced = self._coerce_value(value)
        self._apply_setting(section, key, coerced)
        save_settings(self.settings_path, self.settings)
        await ctx.reply(f"Updated {section}.{key} to {coerced}.")
        if section in {"data", "backtest", "strategy"}:
            await self._update_strategy_list_channel()

    async def set_config_bulk_text(self, ctx: commands.Context, *, updates: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can change settings.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(
                f"Please use #{self.command_channel_name} for config changes."
            )
            return
        if not updates.strip():
            await ctx.reply(
                "Usage: `!set_config_bulk section.key=value` (one per line)"
            )
            return

        parsed, errors = self._parse_bulk_updates(updates)
        if errors:
            message = "Some lines are invalid:\n" + "\n".join(errors[:8])
            if len(errors) > 8:
                message += f"\n...and {len(errors) - 8} more."
            await ctx.reply(message)
            return

        for section, key, value in parsed:
            self._apply_setting(section, key, value)
        save_settings(self.settings_path, self.settings)
        await ctx.reply(f"Updated {len(parsed)} setting(s).")
        await self._update_strategy_list_channel()

    async def config_edit_text(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can change settings.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(
                f"Please use #{self.command_channel_name} for config changes."
            )
            return

        entries = self._flatten_settings()
        if not entries:
            await ctx.reply("No editable settings found.")
            return

        view = ConfigEditorView(self, entries)
        content = "Select a variable to edit.\n" + view._page_summary()
        view.message = await ctx.reply(content, view=view)

    async def status_text(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for status commands"))
            return
        group = self._resolve_channel_group(ctx.channel)
        if group == "backtest":
            await ctx.reply("Backtest status/log channels are disabled.")
            return
        snapshot = self._status_snapshot_for_group(group)
        if snapshot is None:
            await ctx.reply("No status snapshot available yet.")
            return
        channel = await self._get_or_create_group_channel(
            group, self._group_channel_name(group, "status")
        )
        if channel is None:
            await ctx.reply("Status channel not found.")
            return
        title = "Backtest" if group == "backtest" else group.capitalize()
        embed = self._format_status_embed(snapshot, title=title, group=group)
        await self._clear_status_channel(channel)
        await channel.send(embed=embed)
        await ctx.reply(f"Status posted in #{channel.name}.")

    async def graph_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for realtime chart commands"))
            return
        if args.strip():
            await ctx.reply("Usage: `!graph`")
            return
        group = self._resolve_channel_group(ctx.channel)
        runner = self._runner_for_group(group)
        if runner is None or not runner.is_running():
            await ctx.reply("Runner is not active for this channel.")
            return
        payload = runner.latest_payload
        if payload is None or not isinstance(payload.positions, dict):
            await ctx.reply("No realtime data yet.")
            return
        positions = payload.positions
        active: List[Tuple[str, Dict[str, Any]]] = []
        for side in ("long", "short"):
            raw = positions.get(side, {})
            if not isinstance(raw, dict):
                continue
            if not raw.get("active"):
                continue
            if raw.get("entry_time") is None:
                continue
            active.append((side, raw))
        if not active:
            await ctx.reply("No active position to chart.")
            return
        data = runner.get_chart_data(include_live=True)
        if data.empty:
            await ctx.reply("No price data available yet.")
            return
        if "timestamp" in data.columns:
            last_ts: Any = data.iloc[-1]["timestamp"]
        else:
            last_ts = len(data) - 1
        last_price = float(payload.price or 0.0)
        if not last_price and "close" in data.columns:
            last_price = float(data.iloc[-1].get("close", 0.0) or 0.0)

        chart_paths: List[Path] = []
        for side, raw in active:
            trade = {
                "entry_time": raw.get("entry_time"),
                "timestamp": last_ts,
                "entry_price": raw.get("entry_price"),
                "price": last_price,
                "direction": raw.get("direction", 0),
                "rr_tp_price": raw.get("tp_price"),
                "rr_stop_price": raw.get("sl_price"),
            }
            title = f"{payload.symbol} {side.upper()} entry → now"
            with tempfile.NamedTemporaryFile(
                prefix=f"{group}_{side}_graph_",
                suffix=".png",
                delete=False,
            ) as tmp_file:
                chart_path = create_trade_snapshot_chart(
                    data,
                    trade,
                    Path(tmp_file.name),
                    title=title,
                    lookback=0,
                    lookforward=0,
                    max_candles=400,
                )
            if chart_path:
                chart_paths.append(chart_path)
        if not chart_paths:
            await ctx.reply("Failed to build chart.")
            return
        files = [discord.File(path) for path in chart_paths]
        await ctx.reply(files=files)

    async def sim_text(self, ctx: commands.Context, *, args: str = "") -> None:
        await self._realtime_text_command(ctx, group="sim", args=args)

    async def live_text(self, ctx: commands.Context, *, args: str = "") -> None:
        await self._realtime_text_command(ctx, group="live", args=args)

    async def order_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for live orders"))
            return
        if not self._is_live_channel(ctx.channel):
            await ctx.reply("`!order` is only available in live channels.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can place live orders.")
            return

        cfg = self._get_realtime_cfg("live")
        default_symbol = str(cfg.get("symbol", "XRPUSDT")).upper()
        try:
            default_leverage = float(
                cfg.get("leverage", self.settings.backtest.get("leverage", 1.0))
            )
        except (TypeError, ValueError):
            default_leverage = 1.0

        if not args.strip():
            usage = (
                "Usage: `!order long|short usdt_amount [price] [symbol] [leverage]`\n"
                f"Defaults: symbol={default_symbol}, leverage={default_leverage}\n"
                "Examples:\n"
                f"- `!order long 10` (uses current {default_symbol} price)\n"
                f"- `!order short 25 0.63 {default_symbol} 5`"
            )
            account_note = ""
            client, err = self._get_live_client()
            if err:
                account_note = f"\n{err}"
            else:
                available, wallet, price, note = await self._fetch_live_account_snapshot(
                    client, default_symbol
                )
                if note:
                    account_note = f"\n{note}"
                else:
                    available_text = (
                        f"{available:,.2f}" if available is not None else "n/a"
                    )
                    wallet_text = f"{wallet:,.2f}" if wallet is not None else "n/a"
                    price_text = self._fmt_price(price)
                    account_note = (
                        f"\nAvailable USDT: {available_text} (wallet {wallet_text})"
                        f"\nCurrent {default_symbol}: {price_text}"
                    )
            view = LiveOrderStartView(
                self, ctx.author.id, default_symbol, default_leverage
            )
            message = await ctx.reply(f"{usage}{account_note}", view=view)
            view.message = message
            return

        await self._handle_live_order(ctx, args)

    async def close_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for live orders"))
            return
        if not self._is_live_channel(ctx.channel):
            await ctx.reply("`!close` is only available in live channels.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can close positions.")
            return
        if args.strip():
            await ctx.reply("Usage: `!close`")
            return

        client, err = self._get_live_client()
        if err:
            await ctx.reply(err)
            return

        try:
            positions = await asyncio.to_thread(client.get_position_risk)
        except Exception as exc:
            await ctx.reply(f"Failed to fetch positions: {exc}")
            return

        if not isinstance(positions, list):
            await ctx.reply("No position data returned.")
            return

        cfg = self._get_realtime_cfg("live")
        maker_offset_bps = float(cfg.get("maker_offset_bps", 1.0) or 0.0)
        drafts: List[ClosePositionDraft] = []
        lines: List[str] = []

        for item in positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "") or "").upper()
            if not symbol:
                continue
            position_amt = float(item.get("positionAmt", 0.0) or 0.0)
            if position_amt == 0:
                continue
            position_side = str(item.get("positionSide", "BOTH") or "BOTH").upper()
            if position_side not in {"LONG", "SHORT"}:
                position_side = "LONG" if position_amt > 0 else "SHORT"
            qty = abs(position_amt)
            if qty <= 0:
                continue
            side = "SELL" if position_side == "LONG" else "BUY"

            price = None
            payload = self._realtime_payloads.get("live")
            if payload and payload.symbol == symbol and payload.price:
                price = float(payload.price)
            else:
                try:
                    price = float(await asyncio.to_thread(client.get_price, symbol))
                except Exception:
                    price = 0.0
            if price <= 0:
                continue
            order_price = self._apply_maker_offset(price, side, maker_offset_bps)
            draft = ClosePositionDraft(
                symbol=symbol,
                side=side,
                position_side=position_side,
                qty=qty,
                price=order_price,
                maker_offset_bps=maker_offset_bps,
            )
            drafts.append(draft)
            entry_price = item.get("entryPrice", "n/a")
            lines.append(
                f"- {symbol} {position_side} qty {qty:.6g} entry {self._fmt_price(entry_price)}"
            )

        if not drafts:
            await ctx.reply("No open positions to close.")
            return

        view = ClosePositionSelectView(self, drafts, ctx.author.id)
        message = await ctx.reply(
            "Open positions:\n" + "\n".join(lines),
            view=view,
        )
        view.message = message

    async def positions_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for live positions"))
            return
        if not self._is_live_channel(ctx.channel):
            await ctx.reply("`!positions` is only available in live channels.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can view live positions.")
            return
        if args.strip():
            await ctx.reply("Usage: `!positions`")
            return

        client, err = self._get_live_client()
        if err:
            await ctx.reply(err)
            return

        try:
            positions = await asyncio.to_thread(client.get_position_risk)
        except Exception as exc:
            await ctx.reply(f"Failed to fetch positions: {exc}")
            return

        if not isinstance(positions, list):
            await ctx.reply("No position data returned.")
            return

        def fmt_money(value: Any) -> str:
            try:
                return f"{float(value):,.2f}"
            except (TypeError, ValueError):
                return "n/a"

        lines: List[str] = []
        total_unreal = 0.0
        count = 0
        for item in positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "") or "").upper()
            if not symbol:
                continue
            position_amt = float(item.get("positionAmt", 0.0) or 0.0)
            if position_amt == 0:
                continue
            position_side = str(item.get("positionSide", "BOTH") or "BOTH").upper()
            if position_side not in {"LONG", "SHORT"}:
                position_side = "LONG" if position_amt > 0 else "SHORT"
            qty = abs(position_amt)
            entry_price = item.get("entryPrice", "n/a")
            mark_price = item.get("markPrice", "n/a")
            leverage_raw = item.get("leverage", "n/a")
            try:
                leverage_value = float(leverage_raw)
            except (TypeError, ValueError):
                leverage_value = 0.0
            liq_price = item.get("liquidationPrice", "n/a")
            notional = 0.0
            try:
                notional = abs(float(item.get("notional", 0.0) or 0.0))
            except (TypeError, ValueError):
                notional = 0.0
            margin = 0.0
            try:
                margin = float(item.get("isolatedMargin", 0.0) or 0.0)
            except (TypeError, ValueError):
                margin = 0.0
            if margin <= 0:
                try:
                    margin = float(item.get("positionInitialMargin", 0.0) or 0.0)
                except (TypeError, ValueError):
                    margin = 0.0
            if margin <= 0 and leverage_value > 0 and notional > 0:
                margin = notional / leverage_value
            unreal = float(item.get("unRealizedProfit", 0.0) or 0.0)
            total_unreal += unreal
            count += 1
            lev_text = f"{leverage_value:g}" if leverage_value > 0 else str(leverage_raw)
            margin_text = f"{margin:,.2f}" if margin > 0 else "n/a"
            line = (
                f"{symbol} {position_side} | qty {qty:.6g} | entry {self._fmt_price(entry_price)} "
                f"| mark {self._fmt_price(mark_price)} | pnl {unreal:+,.2f} | lev {lev_text}x "
                f"| margin {margin_text} | liq {self._fmt_price(liq_price)}"
            )
            lines.append(line)

        if not lines:
            await ctx.reply("No open positions.")
            return

        embed = discord.Embed(
            title="📌 Live Open Positions",
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="Summary",
            value=f"Positions: {count}\nUnrealized PnL: {fmt_money(total_unreal)}",
            inline=False,
        )

        chunk: List[str] = []
        chunk_len = 0
        for line in lines:
            add_len = len(line) + (1 if chunk else 0)
            if chunk_len + add_len > 950:
                embed.add_field(
                    name="Positions",
                    value="\n".join(chunk),
                    inline=False,
                )
                chunk = []
                chunk_len = 0
            chunk.append(line)
            chunk_len += add_len
        if chunk:
            embed.add_field(
                name="Positions",
                value="\n".join(chunk),
                inline=False,
            )

        embed.set_footer(text=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST"))
        await ctx.reply(embed=embed)

    async def assets_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for live assets"))
            return
        if not self._is_live_channel(ctx.channel):
            await ctx.reply("`!assets` is only available in live channels.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can view live assets.")
            return
        if args.strip():
            await ctx.reply("Usage: `!assets`")
            return

        futures_client, err = self._get_live_client()
        if err:
            await ctx.reply(err)
            return

        spot_client, spot_err = self._get_spot_client()
        if spot_err:
            await ctx.reply(spot_err)
            return

        futures_account = None
        spot_account = None
        prices = None

        try:
            futures_account = await asyncio.to_thread(futures_client.get_account)
        except Exception as exc:
            await ctx.reply(f"Failed to fetch futures account: {exc}")
            return

        try:
            spot_account = await asyncio.to_thread(spot_client.get_account)
        except Exception as exc:
            await ctx.reply(f"Failed to fetch spot account: {exc}")
            return

        try:
            prices = await asyncio.to_thread(spot_client.get_prices)
        except Exception:
            prices = None

        price_map: Dict[str, float] = {}
        if isinstance(prices, list):
            for item in prices:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "") or "").upper()
                if not symbol:
                    continue
                try:
                    price_map[symbol] = float(item.get("price", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue

        def fmt_money(value: Any) -> str:
            try:
                return f"{float(value):,.2f}"
            except (TypeError, ValueError):
                return "n/a"

        futures_text = "n/a"
        if isinstance(futures_account, dict):
            total_wallet = futures_account.get("totalWalletBalance")
            total_margin = futures_account.get("totalMarginBalance")
            total_unreal = futures_account.get("totalUnrealizedProfit")
            available = futures_account.get("availableBalance")
            futures_text = (
                f"Wallet: {fmt_money(total_wallet)} USDT\n"
                f"Available: {fmt_money(available)} USDT\n"
                f"Margin: {fmt_money(total_margin)} USDT\n"
                f"Unrealized: {fmt_money(total_unreal)} USDT"
            )

        spot_lines: List[Tuple[float, str]] = []
        spot_total = 0.0
        total_assets = 0
        balances = spot_account.get("balances") if isinstance(spot_account, dict) else None
        if isinstance(balances, list):
            for bal in balances:
                if not isinstance(bal, dict):
                    continue
                asset = str(bal.get("asset", "") or "").upper()
                try:
                    free = float(bal.get("free", 0.0) or 0.0)
                except (TypeError, ValueError):
                    free = 0.0
                try:
                    locked = float(bal.get("locked", 0.0) or 0.0)
                except (TypeError, ValueError):
                    locked = 0.0
                total = free + locked
                if total <= 0:
                    continue
                total_assets += 1
                est_usdt = None
                if asset == "USDT":
                    est_usdt = total
                else:
                    symbol = f"{asset}USDT"
                    price = price_map.get(symbol)
                    if price:
                        est_usdt = total * price
                est_text = fmt_money(est_usdt) if est_usdt is not None else "n/a"
                if est_usdt is not None:
                    spot_total += est_usdt
                sort_value = est_usdt if est_usdt is not None else 0.0
                spot_lines.append(
                    (
                        sort_value,
                        f"{asset}: {total:.6g} (free {free:.6g}) | ≈ {est_text} USDT",
                    )
                )

        if spot_lines:
            spot_lines.sort(key=lambda item: item[0], reverse=True)
            spot_display = [line for _, line in spot_lines]
        else:
            spot_display = ["No spot balances."]

        embed = discord.Embed(
            title="💼 Account Assets",
            color=discord.Color.blue(),
        )
        embed.add_field(name="Futures (USDT)", value=futures_text, inline=False)
        embed.add_field(
            name="Spot Summary",
            value=f"Assets: {total_assets}\nEstimated USDT: {fmt_money(spot_total)}",
            inline=False,
        )

        chunk: List[str] = []
        chunk_len = 0
        for line in spot_display:
            add_len = len(line) + (1 if chunk else 0)
            if chunk_len + add_len > 950:
                embed.add_field(
                    name="Spot Balances",
                    value="\n".join(chunk),
                    inline=False,
                )
                chunk = []
                chunk_len = 0
            chunk.append(line)
            chunk_len += add_len
        if chunk:
            embed.add_field(
                name="Spot Balances",
                value="\n".join(chunk),
                inline=False,
            )

        embed.set_footer(text=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST"))
        await ctx.reply(embed=embed)

    async def _handle_live_order(
        self,
        target: Union[discord.Interaction, commands.Context],
        args: str,
        *,
        modal_fields: Optional[Dict[str, str]] = None,
    ) -> None:
        async def _send(message: str, *, ephemeral: bool = False) -> None:
            if isinstance(target, discord.Interaction):
                channel = getattr(target, "channel", None)
                if self._interaction_is_expired(target):
                    if channel is not None:
                        await channel.send(message)
                    return
                try:
                    if target.response.is_done():
                        await target.followup.send(message, ephemeral=ephemeral)
                    else:
                        await target.response.send_message(message, ephemeral=ephemeral)
                    return
                except (discord.NotFound, discord.errors.NotFound, discord.HTTPException) as exc:
                    if getattr(exc, "code", None) not in {None, 10062}:
                        raise
                    try:
                        await target.followup.send(message, ephemeral=ephemeral)
                        return
                    except Exception:
                        if channel is not None:
                            await channel.send(message)
            else:
                await target.reply(message)

        async def _send_with_view(
            message: str,
            view: View,
            *,
            ephemeral: bool = False,
        ) -> None:
            if isinstance(target, discord.Interaction):
                channel = getattr(target, "channel", None)
                if self._interaction_is_expired(target):
                    msg = await channel.send(message, view=view) if channel else None
                else:
                    try:
                        if target.response.is_done():
                            msg = await target.followup.send(
                                message, view=view, ephemeral=ephemeral
                            )
                        else:
                            await target.response.send_message(
                                message, view=view, ephemeral=ephemeral
                            )
                            msg = await target.original_response()
                    except (discord.NotFound, discord.errors.NotFound, discord.HTTPException) as exc:
                        if getattr(exc, "code", None) not in {None, 10062}:
                            raise
                        try:
                            msg = await target.followup.send(
                                message, view=view, ephemeral=ephemeral
                            )
                        except Exception:
                            msg = await channel.send(message, view=view) if channel else None
            else:
                msg = await target.reply(message, view=view)
            view.message = msg

        cfg = self._get_realtime_cfg("live")
        default_symbol = str(cfg.get("symbol", "XRPUSDT")).upper()
        try:
            default_leverage = float(
                cfg.get("leverage", self.settings.backtest.get("leverage", 1.0))
            )
        except (TypeError, ValueError):
            default_leverage = 1.0

        def _looks_like_symbol(value: str) -> bool:
            return any(ch.isalpha() for ch in value)

        def _extract_float(text: str) -> Optional[float]:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None

        amount: Optional[float] = None
        price: Optional[float] = None
        symbol: Optional[str] = None
        leverage: Optional[float] = None
        side = ""
        position_side = ""

        if modal_fields is not None:
            side_text = (modal_fields.get("side") or "").strip().lower()
            if side_text in {"buy", "long", "l", "b"}:
                side = "BUY"
                position_side = "LONG"
            elif side_text in {"sell", "short", "s"}:
                side = "SELL"
                position_side = "SHORT"
            else:
                await _send("Side must be long/short (or buy/sell).", ephemeral=True)
                return

            amount_text = (modal_fields.get("usdt_amount") or "").strip()
            if not amount_text:
                await _send("USDT amount must be a number.", ephemeral=True)
                return
            amount = _extract_float(amount_text)
            if amount is None:
                await _send("USDT amount must be a number.", ephemeral=True)
                return

            price_text = (modal_fields.get("price") or "").strip()
            if price_text:
                price = _extract_float(price_text)
                if price is None:
                    await _send("Price must be a number.", ephemeral=True)
                    return

            symbol_text = (modal_fields.get("symbol") or "").strip()
            if symbol_text and _looks_like_symbol(symbol_text):
                symbol = symbol_text.replace("/", "").replace("-", "").upper()

            leverage_text = (modal_fields.get("leverage") or "").strip()
            if leverage_text:
                leverage = _extract_float(leverage_text)
                if leverage is None:
                    await _send("Leverage must be a number.", ephemeral=True)
                    return
        else:
            tokens = shlex.split(args or "")
            if not tokens:
                await _send(
                    "Usage: `!order long|short usdt_amount [price] [symbol] [leverage]`",
                    ephemeral=True,
                )
                return

            if len(tokens) < 2:
                await _send(
                    "Usage: `!order long|short usdt_amount [price] [symbol] [leverage]`",
                    ephemeral=True,
                )
                return

            side_token = tokens[0].lower()
            if side_token in {"buy", "long", "l", "b"}:
                side = "BUY"
                position_side = "LONG"
            elif side_token in {"sell", "short", "s"}:
                side = "SELL"
                position_side = "SHORT"
            else:
                await _send("Side must be long/short (or buy/sell).", ephemeral=True)
                return

            side_tokens = {
                "buy",
                "sell",
                "long",
                "short",
                "l",
                "s",
                "b",
            }
            leverage_tokens = {"leverage", "lev", "레버리지"}
            amount_tokens = {"usdt", "usd", "amount", "원금", "증거금"}
            skip_indices: Set[int] = set()

            for idx, token in enumerate(tokens):
                if token.lower() in side_tokens:
                    skip_indices.add(idx)

            for idx, token in enumerate(tokens):
                if idx in skip_indices:
                    continue
                text = token.strip()
                if not text:
                    continue
                lower = text.lower()
                if lower.endswith("x") and leverage is None:
                    parsed = _extract_float(lower[:-1])
                    if parsed is not None:
                        leverage = parsed
                        continue
                if any(key in lower for key in leverage_tokens) and leverage is None:
                    parsed = _extract_float(text)
                    if parsed is None and idx + 1 < len(tokens):
                        parsed = _extract_float(tokens[idx + 1])
                        if parsed is not None:
                            skip_indices.add(idx + 1)
                    if parsed is not None:
                        leverage = parsed
                        continue
                if any(key in lower for key in amount_tokens) and amount is None:
                    parsed = _extract_float(text)
                    if parsed is None and idx + 1 < len(tokens):
                        parsed = _extract_float(tokens[idx + 1])
                        if parsed is not None:
                            skip_indices.add(idx + 1)
                    if parsed is not None:
                        amount = parsed
                        continue

            for idx, token in enumerate(tokens):
                if idx in skip_indices:
                    continue
                text = token.strip()
                if not text:
                    continue
                if _looks_like_symbol(text) and symbol is None:
                    lower = text.lower()
                    if lower not in amount_tokens and not any(
                        key in lower for key in leverage_tokens
                    ):
                        symbol = text.replace("/", "").replace("-", "").upper()
                        continue
                parsed = _extract_float(text)
                if parsed is None:
                    continue
                if amount is None:
                    amount = parsed
                    continue
                if price is None:
                    price = parsed
                    continue
                if leverage is None:
                    leverage = parsed
                    continue

        if amount is None:
            await _send("USDT amount must be a number.", ephemeral=True)
            return
        usdt_amount = amount
        if usdt_amount <= 0:
            await _send("USDT amount must be greater than 0.", ephemeral=True)
            return

        if symbol is None:
            symbol = default_symbol
        if leverage is None:
            leverage = default_leverage
        if leverage <= 0:
            await _send("Leverage must be greater than 0.", ephemeral=True)
            return
        if price is not None and price <= 0:
            await _send("Price must be greater than 0.", ephemeral=True)
            return

        client, err = self._get_live_client()
        if err:
            await _send(err, ephemeral=True)
            return

        available, wallet, current_price, note = await self._fetch_live_account_snapshot(
            client, symbol
        )
        if current_price is None:
            await _send(note or "Failed to fetch current price.", ephemeral=True)
            return

        if price is None:
            price = current_price

        maker_offset_bps = float(cfg.get("maker_offset_bps", 1.0) or 0.0)
        order_price = self._apply_maker_offset(price, side, maker_offset_bps)

        tick_size = 0.0
        step_size = 0.0
        min_notional = None
        min_price = None
        max_price = None
        try:
            filters = await asyncio.to_thread(client.get_symbol_filters, symbol)
            tick_size = filters.tick_size
            step_size = filters.step_size
            min_notional = filters.min_notional
            min_price, max_price = self._price_bounds(
                float(current_price), filters
            )
        except Exception:
            pass

        raw_order_price = order_price
        order_price = RealtimeRunner._round_price_with_bounds(
            order_price, tick_size, side, min_price, max_price
        )
        if order_price <= 0:
            await _send("Price invalid after rounding.", ephemeral=True)
            return

        notional = usdt_amount * float(leverage)
        qty = notional / order_price
        qty = RealtimeRunner._round_step(qty, step_size)
        if qty <= 0:
            await _send("Order size too small after symbol rounding.", ephemeral=True)
            return
        rounded_notional = qty * order_price
        if min_notional and rounded_notional < min_notional:
            await _send(
                f"Order notional must be at least {min_notional:,.4g} USDT. "
                f"Current notional: {rounded_notional:,.4g} USDT.",
                ephemeral=True,
            )
            return

        draft = LiveOrderDraft(
            symbol=symbol,
            side=side,
            position_side=position_side,
            leverage=float(leverage),
            usdt_amount=usdt_amount,
            price=order_price,
            qty=qty,
            reduce_only=False,
            maker_offset_bps=maker_offset_bps,
        )

        warnings: List[str] = []
        if (
            (min_price is not None or max_price is not None)
            and abs(order_price - raw_order_price) > 1e-12
        ):
            bound_parts: List[str] = []
            if min_price is not None:
                bound_parts.append(f"min {self._fmt_price(min_price)}")
            if max_price is not None:
                bound_parts.append(f"max {self._fmt_price(max_price)}")
            bounds_text = ", ".join(bound_parts)
            warnings.append(
                f"Note: price adjusted to exchange bounds ({bounds_text})."
            )
        if available is not None and usdt_amount > available:
            warnings.append(
                f"Warning: amount {usdt_amount:,.2f} exceeds available "
                f"{available:,.2f} USDT."
            )

        available_text = f"{available:,.2f}" if available is not None else "n/a"
        wallet_text = f"{wallet:,.2f}" if wallet is not None else "n/a"
        price_text = self._fmt_price(current_price if current_price is not None else price)
        warnings_text = "\n".join(warnings) if warnings else ""
        confirm_text = (
            "Live futures order preview\n"
            f"- Symbol: {symbol}\n"
            f"- Side: {position_side}\n"
            f"- Leverage: {float(leverage):.2f}x\n"
            f"- Amount (USDT, margin): {usdt_amount:,.2f}\n"
            f"- Notional (USDT): {notional:,.2f}\n"
            f"- Qty: {qty:.6g}\n"
            f"- Order price (maker): {self._fmt_price(order_price)}\n"
            f"- Current price: {price_text}\n"
            f"- Available USDT: {available_text} (wallet {wallet_text})\n"
            "- Margin: ISOLATED\n"
            "- Mode: HEDGE"
            + (f"\n{warnings_text}" if warnings_text else "")
        )
        if note:
            confirm_text = f"{confirm_text}\nNote: {note}"
        view = LiveOrderConfirmView(
            self,
            target.user.id if isinstance(target, discord.Interaction) else target.author.id,
            draft,
        )
        await _send_with_view(confirm_text, view, ephemeral=isinstance(target, discord.Interaction))

    async def _realtime_text_command(
        self, ctx: commands.Context, group: str, *, args: str = ""
    ) -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for realtime commands"))
            return

        tokens = shlex.split(args or "")
        if tokens:
            first = tokens[0].lower()
            if first in {"api", "apikey", "api_key", "key", "secret", "apisecret", "api_secret", "credentials", "creds"}:
                if group != "live":
                    await ctx.reply("API 키/시크릿 설정은 live에서만 지원돼요.")
                    return
                if not self._is_guild_owner_ctx(ctx):
                    await ctx.reply("Only the server owner can change live API credentials.")
                    return
                cfg = self._get_realtime_cfg("live")
                api_key = str(cfg.get("api_key", "") or "")
                api_secret = str(cfg.get("api_secret", "") or "")

                def mask(value: str) -> str:
                    if not value:
                        return "MISSING"
                    if len(value) <= 6:
                        return value[0] + ("*" * max(len(value) - 2, 1)) + value[-1]
                    return f"{value[:4]}...{value[-4:]}"

                if first in {"api", "credentials", "creds"}:
                    if len(tokens) == 1:
                        await ctx.reply(
                            f"Live API 설정 상태\n- api_key: {mask(api_key)}\n- api_secret: {mask(api_secret)}\n"
                            "사용법: `!live api <API_KEY> <API_SECRET>`"
                        )
                        return
                    if len(tokens) < 3:
                        await ctx.reply("사용법: `!live api <API_KEY> <API_SECRET>`")
                        return
                    cfg["api_key"] = tokens[1].strip()
                    cfg["api_secret"] = tokens[2].strip()
                elif first in {"apikey", "api_key", "key"}:
                    if len(tokens) < 2:
                        await ctx.reply("사용법: `!live apikey <API_KEY>`")
                        return
                    cfg["api_key"] = tokens[1].strip()
                else:
                    if len(tokens) < 2:
                        await ctx.reply("사용법: `!live apisecret <API_SECRET>`")
                        return
                    cfg["api_secret"] = tokens[1].strip()

                save_settings(self.settings_path, self.settings)
                runner = self.live_runner
                running = bool(runner and getattr(runner, "is_running", lambda: False)())
                if running:
                    self._stop_realtime_runner("live")
                    err = self._start_realtime_runner("live")
                    if err:
                        await ctx.reply(f"Live API 정보를 저장했지만 재시작 실패: {err}")
                        return
                    await ctx.reply(
                        f"Live API 정보를 저장했어요. (runner 재시작 완료)\n"
                        f"- api_key: {mask(str(cfg.get('api_key','') or ''))}\n"
                        f"- api_secret: {mask(str(cfg.get('api_secret','') or ''))}"
                    )
                else:
                    await ctx.reply(
                        f"Live API 정보를 저장했어요. (다음 시작부터 적용)\n"
                        f"- api_key: {mask(str(cfg.get('api_key','') or ''))}\n"
                        f"- api_secret: {mask(str(cfg.get('api_secret','') or ''))}"
                    )
                return
            if first in {"asset", "balance", "initial_balance", "capital"}:
                if group != "sim":
                    await ctx.reply("자산 설정은 sim에서만 지원돼요.")
                    return
                if not self._is_guild_owner_ctx(ctx):
                    await ctx.reply("Only the server owner can change sim assets.")
                    return
                cfg = self._get_realtime_cfg("sim")
                current = cfg.get("initial_balance", self.settings.backtest.get("initial_balance"))
                if len(tokens) < 2:
                    await ctx.reply(
                        f"현재 sim 초기자산: {current}\n사용법: `!sim asset 10000`"
                    )
                    return
                raw_value = tokens[1].replace(",", "")
                try:
                    new_value = float(raw_value)
                except ValueError:
                    await ctx.reply("자산 값이 숫자가 아니에요. 예: `!sim asset 10000`")
                    return
                if new_value <= 0:
                    await ctx.reply("자산 값은 0보다 커야 해요.")
                    return
                cfg["initial_balance"] = new_value
                save_settings(self.settings_path, self.settings)
                runner = self.sim_runner
                running = bool(runner and getattr(runner, "is_running", lambda: False)())
                if running:
                    self._stop_realtime_runner("sim")
                    err = self._start_realtime_runner("sim")
                    if err:
                        await ctx.reply(
                            f"sim 초기자산을 {new_value}로 저장했지만 재시작 실패: {err}"
                        )
                        return
                    await ctx.reply(
                        f"sim 초기자산을 {new_value}로 설정했어요. (runner 재시작 완료)"
                    )
                else:
                    await ctx.reply(
                        f"sim 초기자산을 {new_value}로 설정했어요. (다음 시작부터 적용)"
                    )
                return
            if first in {"interval", "timeframe", "tf", "bar"}:
                if not self._is_guild_owner_ctx(ctx):
                    await ctx.reply("Only the server owner can change intervals.")
                    return
                cfg = self._get_realtime_cfg(group)
                current = str(cfg.get("interval", "") or "")
                if len(tokens) < 2:
                    await ctx.reply(
                        f"현재 {group} interval: {current or 'n/a'}\n"
                        f"사용법: `!{group} interval 5m`"
                    )
                    return
                value = tokens[1].strip()
                if value not in self._supported_intervals():
                    await ctx.reply(
                        "지원되지 않는 interval이에요.\n"
                        f"가능한 값: {', '.join(self._supported_intervals())}"
                    )
                    return
                cfg["interval"] = value
                save_settings(self.settings_path, self.settings)
                runner = self._runner_for_group(group)
                running = bool(runner and getattr(runner, "is_running", lambda: False)())
                if running:
                    self._stop_realtime_runner(group)
                    err = self._start_realtime_runner(group)
                    if err:
                        await ctx.reply(f"{group} interval을 {value}로 저장했지만 재시작 실패: {err}")
                        return
                    await ctx.reply(f"{group} interval을 {value}로 설정했어요. (runner 재시작 완료)")
                else:
                    await ctx.reply(f"{group} interval을 {value}로 설정했어요. (다음 시작부터 적용)")
                return
            if first in {"symbol", "ticker", "coin", "pair"}:
                if not self._is_guild_owner_ctx(ctx):
                    await ctx.reply("Only the server owner can change symbols.")
                    return
                cfg = self._get_realtime_cfg(group)
                current = str(cfg.get("symbol", "") or "").upper()
                if len(tokens) < 2:
                    await ctx.reply(
                        f"현재 {group} 심볼: {current or 'n/a'}\n"
                        f"사용법: `!{group} symbol XRPUSDT`"
                    )
                    return
                raw = tokens[1].strip()
                symbol = raw.replace("/", "").replace("-", "").upper()
                if not symbol:
                    await ctx.reply("심볼이 비어있어요. 예: `!sim symbol XRPUSDT`")
                    return
                cfg["symbol"] = symbol
                save_settings(self.settings_path, self.settings)
                runner = self._runner_for_group(group)
                running = bool(runner and getattr(runner, "is_running", lambda: False)())
                if running:
                    self._stop_realtime_runner(group)
                    err = self._start_realtime_runner(group)
                    if err:
                        await ctx.reply(f"{group} 심볼을 {symbol}로 저장했지만 재시작 실패: {err}")
                        return
                    await ctx.reply(f"{group} 심볼을 {symbol}로 설정했어요. (runner 재시작 완료)")
                else:
                    await ctx.reply(f"{group} 심볼을 {symbol}로 설정했어요. (다음 시작부터 적용)")
                return

        action = self._parse_realtime_action(args or "")
        if action is None:
            await ctx.reply(
                "Usage: `!sim on|off|status|restart|asset|interval|symbol` "
                "or `!live on|off|status|restart|api|interval|symbol`"
            )
            return

        if action != "status" and not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can start/stop realtime runners.")
            return

        if action == "status":
            cfg = self._get_realtime_cfg(group)
            enabled = bool(cfg.get("enabled", False))
            runner = self._runner_for_group(group)
            running = bool(runner and runner._task and not runner._task.done())
            symbol = str(cfg.get("symbol", "") or "").upper() or "UNKNOWN"
            interval = str(cfg.get("interval", "") or "")
            auto_trade = bool(cfg.get("auto_trade", False))
            maker_offset = cfg.get("maker_offset_bps", "n/a")
            await ctx.reply(
                f"{group.upper()} 상태: enabled={enabled} | running={running} | "
                f"auto_trade={auto_trade} | {symbol} {interval} | maker_offset_bps={maker_offset}"
            )
            return

        if action == "off":
            self._stop_realtime_runner(group)
            self._set_realtime_enabled(group, False)
            await ctx.reply(f"{group.upper()} runner stopped.")
            return

        if action == "restart":
            self._stop_realtime_runner(group)
            self._set_realtime_enabled(group, True)
            err = self._start_realtime_runner(group)
            if err:
                await ctx.reply(f"{group.upper()} restart failed: {err}")
                return
            await ctx.reply(f"{group.upper()} runner restarted.")
            return

        if action == "on":
            self._set_realtime_enabled(group, True)
            err = self._start_realtime_runner(group)
            if err:
                await ctx.reply(f"{group.upper()} start failed: {err}")
                return
            await ctx.reply(f"{group.upper()} runner started.")
            return

    def _setting_profile_path(self, name: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name.strip())
        if not safe:
            safe = "profile"
        if not safe.endswith(".yaml") and not safe.endswith(".yml"):
            safe += ".yaml"
        return self.setting_profiles_dir / safe

    def _load_setting_profile(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._setting_profile_path(name)
        if not path.exists():
            return None
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        return raw

    def _list_setting_profiles(self) -> List[str]:
        self.setting_profiles_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            p.name for p in self.setting_profiles_dir.glob("*.y*ml") if p.is_file()
        )

    def _format_setting_profile_label(self, name: str) -> str:
        profile = self._load_setting_profile(name)
        display = name
        description = None
        if isinstance(profile, dict):
            meta = profile.get("meta")
            if isinstance(meta, dict):
                meta_name = meta.get("name")
                meta_description = meta.get("description")
                if isinstance(meta_name, str) and meta_name.strip():
                    display = meta_name.strip()
                if isinstance(meta_description, str) and meta_description.strip():
                    description = meta_description.strip()
        label = display
        if description:
            label = f"{label} ({description})"
        if display != name:
            label = f"{label} [{name}]"
        return label

    @staticmethod
    def _parse_saved_at(value: str) -> Optional[datetime]:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1]
            try:
                return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _format_setting_profile_updated_at(self, name: str) -> Optional[str]:
        path = self._setting_profile_path(name)
        if not path.exists():
            return None
        profile = self._load_setting_profile(name)
        saved_at = None
        if isinstance(profile, dict):
            meta = profile.get("meta")
            if isinstance(meta, dict):
                saved_at = meta.get("saved_at")
        updated_at: Optional[datetime] = None
        if isinstance(saved_at, str):
            updated_at = self._parse_saved_at(saved_at)
        if updated_at is None:
            try:
                updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                return None
        return updated_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    def _build_settings_payload(
        self, name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "data": self.settings.data,
            "backtest": self.settings.backtest,
            "strategy": self.settings.strategy,
        }
        if name or description:
            payload["meta"] = {
                "name": name,
                "description": description,
                "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        return payload

    def _save_setting_profile(
        self,
        name: str,
        description: Optional[str],
        overwrite: bool,
    ) -> tuple[Optional[Path], Optional[str]]:
        path = self._setting_profile_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            return None, "That profile already exists. Use overwrite to replace it."
        payload = self._build_settings_payload(name=name, description=description)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path, None

    async def _apply_setting_profile(self, name: str) -> tuple[bool, str]:
        profile = self._load_setting_profile(name)
        if profile is None:
            return False, "Profile not found."
        data = profile.get("data")
        backtest = profile.get("backtest")
        strategy = profile.get("strategy")
        if not isinstance(data, dict) or not isinstance(backtest, dict) or not isinstance(strategy, dict):
            return False, "Profile is missing required sections (data/backtest/strategy)."
        self.settings.data = copy.deepcopy(data)
        self.settings.backtest = copy.deepcopy(backtest)
        self.settings.strategy = copy.deepcopy(strategy)
        save_settings(self.settings_path, self.settings)
        await self._update_strategy_list_channel()

        restart_lines: List[str] = []
        for group in ("sim", "live"):
            runner = self._runner_for_group(group)
            running = bool(runner and getattr(runner, "is_running", lambda: False)())
            if not running:
                continue
            self._stop_realtime_runner(group)
            err = self._start_realtime_runner(group)
            if err:
                restart_lines.append(f"{group} restart failed: {err}")
            else:
                restart_lines.append(f"{group} runner restarted.")

        message = f"Applied settings profile `{name}`."
        if restart_lines:
            message = message + "\n" + "\n".join(restart_lines)
        else:
            message = message + "\nNo runners were running; changes apply on next start."
        return True, message

    async def setting_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can change settings.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(self._command_channel_notice("for setting commands"))
            return
        tokens = shlex.split(args) if args else []
        if not tokens:
            await ctx.reply(
                "Usage: `!setting show [name]`, `!setting edit section.key value`, "
                "`!setting list`, `!setting save`, `!setting apply [name]`, "
                "`!setting upload name`, "
                "`!setting download`, `!setting delete name`"
            )
            return
        action = tokens[0].lower()
        if action == "list":
            profiles = self._list_setting_profiles()
            if not profiles:
                await ctx.reply("No saved setting profiles.")
                return
            lines = []
            for name in profiles:
                label = self._format_setting_profile_label(name)
                updated = self._format_setting_profile_updated_at(name)
                if updated:
                    lines.append(f"- {label} (updated {updated})")
                else:
                    lines.append(f"- {label}")
            await ctx.reply("Saved profiles:\n" + "\n".join(lines))
            return
        if action == "save":
            view = SettingSaveChoiceView(self, ctx.author.id)
            message = await ctx.reply(
                "Save current settings: choose new or overwrite.",
                view=view,
            )
            view.message = message
            return
        if action == "apply":
            profiles = self._list_setting_profiles()
            if not profiles:
                await ctx.reply("No saved setting profiles.")
                return
            if len(tokens) >= 2:
                name = tokens[1]
                ok, message = await self._apply_setting_profile(name)
                await ctx.reply(message)
                return
            view = SettingApplyView(self, profiles, ctx.author.id)
            message = await ctx.reply(
                "Select a profile to apply.",
                view=view,
            )
            view.message = message
            return
        if action == "upload":
            if len(tokens) < 2:
                await ctx.reply("Usage: `!setting upload profile_name`")
                return
            name = tokens[1]
            path, err = self._save_setting_profile(
                name, description=None, overwrite=True
            )
            if err:
                await ctx.reply(err)
                return
            await ctx.reply(f"Saved current settings to `{path.name}`.")
            return
        if action == "download":
            payload = self._build_settings_payload()
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(yaml.safe_dump(payload, sort_keys=False))
                tmp_path = Path(tmp.name)
            try:
                await ctx.reply(files=[discord.File(tmp_path)])
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return
        if action == "delete":
            if len(tokens) < 2:
                await ctx.reply("Usage: `!setting delete profile_name`")
                return
            path = self._setting_profile_path(tokens[1])
            if not path.exists():
                await ctx.reply("Profile not found.")
                return
            try:
                path.unlink()
            except Exception:
                await ctx.reply("Failed to delete profile.")
                return
            await ctx.reply(f"Deleted profile `{path.name}`.")
            return
        if action == "show":
            if len(tokens) >= 2:
                profile = self._load_setting_profile(tokens[1])
                if profile is None:
                    await ctx.reply("Profile not found.")
                    return
                content = yaml.safe_dump(profile, sort_keys=False)
                lines = content.splitlines()
                view = TextPagerView(
                    f"Settings profile: {tokens[1]}",
                    lines,
                    page_size=30,
                    code_block="yaml",
                )
                message = await ctx.reply(view.page_content(), view=view)
                view.message = message
                return
            entries = self._flatten_settings()
            lines = [f"- {key}: {value}" for key, value in entries]
            view = TextPagerView("Current settings", lines, page_size=30)
            message = await ctx.reply(view.page_content(), view=view)
            view.message = message
            return
        if action == "edit":
            if len(tokens) < 3:
                await ctx.reply("Usage: `!setting edit section.key value`")
                return
            raw_key = tokens[1]
            raw_value = " ".join(tokens[2:])
            if "." not in raw_key:
                await ctx.reply("Use section.key format (data/backtest/strategy).")
                return
            section, key = raw_key.split(".", 1)
            if section not in {"data", "backtest", "strategy"}:
                await ctx.reply("Section must be one of: data, backtest, strategy.")
                return
            index = None
            base_key = key
            if "[" in key and key.endswith("]"):
                base_key, index_part = key.rsplit("[", 1)
                index_text = index_part[:-1]
                if not index_text.isdigit():
                    await ctx.reply("List index must be a number like [1].")
                    return
                index = int(index_text) - 1
            coerced = self._coerce_value(raw_value)
            applied = self._apply_setting_value_to(
                self.settings, section, base_key, coerced, index
            )
            if not applied:
                await ctx.reply("Failed to apply setting (bad key or index).")
                return
            save_settings(self.settings_path, self.settings)
            await ctx.reply(f"Updated {section}.{key} to {coerced}.")
            await self._update_strategy_list_channel()
            return
        await ctx.reply(
            "Unknown action. Use `show`, `edit`, `list`, `save`, "
            "`apply`, `upload`, `download`, or `delete`."
        )

    def _help_lines(self) -> List[str]:
        lines = [
            "General",
            "- `!help` : Show this help (paged).",
            "- `!sync` : Sync slash commands for this server (owner only).",
            "",
            "Data",
            "- `!download SYMBOL INTERVAL START END [MARKET]` : Download data.",
            "  e.g. `!download BTCUSDT 5m 2024-01-01 2024-02-01 futures`",
            "- `!delete_data [pattern]` : Delete cached data files.",
            "",
            "Backtest",
            "- `!testrun [options]` : Run backtest (see prompt for options).",
            "- `!diff [options]` : Compare entry diff.",
            "- `!grid [options]` : Grid search.",
            "- `!comp [options]` : Compare strategies.",
            "- `!costom [options]` : Custom batch run.",
            "- `!costom2 [--data FILE]` : Equity + MDD + monthly stats chart.",
            "- `!costom3 [options]` : Overlay grid results + monthly best label.",
            "- `!costom4 [--data FILE]` : Monthly return stability + dispersion stats.",
            "- `!chart [options]` : Render charts.",
            "",
            "Strategy",
            "- `!stge set [strategy_name]` : Set active strategy.",
            "",
            "Config",
            "- `!set_config section key value` : Update config values.",
            "- `!set_config_bulk` : Bulk update (one per line).",
            "- `!conf` : Interactive config editor (paged).",
            "",
            "Settings",
            "- `!setting show [name]` : Show current or saved profile.",
            "- `!setting edit section.key value` : Edit a setting.",
            "- `!setting list` : List profiles.",
            "- `!setting save` : Save profile (new/overwrite).",
            "- `!setting apply [name]` : Apply a saved profile.",
            "- `!setting upload name` : Save profile by name.",
            "- `!setting download` : Download current settings.",
            "- `!setting delete name` : Delete profile.",
            "",
            "Status",
            "- `!status` : Post status embed to sim/live status channels.",
            "",
            "Realtime charts",
            "- `!graph` : Chart from entry to now for active positions.",
            "",
            "Fun",
            "- `!wordle` : Start a 5-letter wordle game in the chat channel.",
            "- `!wordle GUESS` : Submit a guess.",
            "",
            "Realtime (sim)",
            "- `!sim on|off|restart|status` : Control sim runner.",
            "- `!sim asset [amount]` : View/set sim initial balance.",
            "- `!sim interval [5m]` : View/set sim interval.",
            "- `!sim symbol [XRPUSDT]` : View/set sim symbol.",
            "",
            "Realtime (live)",
            "- `!live on|off|restart|status` : Control live runner.",
            "- `!live api [KEY SECRET]` : View/set API key & secret.",
            "- `!live apikey [KEY]` : Set API key only.",
            "- `!live apisecret [SECRET]` : Set API secret only.",
            "- `!live interval [5m]` : View/set live interval.",
            "- `!live symbol [BTCUSDT]` : View/set live symbol.",
            "- `!order buy|sell qty [price] [symbol] [leverage] [reduce]` : Place live order.",
            "- `!close` : Close an open live position (interactive).",
            "- `!positions` : Show live open futures positions.",
            "- `!assets` : Show futures + spot balances.",
        ]
        return lines

    def _wordle_feedback(self, word: str, guess: str) -> str:
        word = word.lower()
        guess = guess.lower()
        result = ["."] * len(word)
        remaining: Dict[str, int] = {}
        for idx, (w_char, g_char) in enumerate(zip(word, guess)):
            if g_char == w_char:
                result[idx] = "G"
            else:
                remaining[w_char] = remaining.get(w_char, 0) + 1
        for idx, g_char in enumerate(guess):
            if result[idx] == "G":
                continue
            if remaining.get(g_char, 0) > 0:
                result[idx] = "Y"
                remaining[g_char] -= 1
        return "".join(result)

    def _wordle_board(self, game: WordleGame) -> str:
        lines = []
        for index, guess in enumerate(game.guesses, start=1):
            feedback = self._wordle_feedback(game.word, guess)
            lines.append(f"{index}. {guess.upper()} {feedback}")
        if not lines:
            return "No guesses yet."
        return "\n".join(lines)

    def _get_or_start_wordle(self, channel_id: int) -> WordleGame:
        game = self._wordle_games.get(channel_id)
        if game is None or game.completed:
            word = random.choice(WORDLE_WORDS)
            game = WordleGame(word=word, guesses=[], max_attempts=6, completed=False)
            self._wordle_games[channel_id] = game
        return game

    async def wordle_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_chat_channel(ctx.channel):
            await ctx.reply(self._chat_channel_notice())
            return

        tokens = shlex.split(args) if args else []
        game = self._wordle_games.get(ctx.channel.id)

        if not tokens:
            game = self._get_or_start_wordle(ctx.channel.id)
            board = self._wordle_board(game)
            await ctx.reply(
                "Wordle started. Guess a 5-letter word with `!wordle GUESS`.\n"
                "Legend: G=correct spot, Y=present, .=absent.\n"
                f"Attempts left: {game.attempts_left}\n{board}"
            )
            return

        guess = tokens[0].strip().lower()
        if len(guess) != 5 or not guess.isalpha():
            await ctx.reply("Guess must be exactly 5 letters. Usage: `!wordle GUESS`")
            return

        if game is None or game.completed:
            game = self._get_or_start_wordle(ctx.channel.id)

        if game.attempts_left <= 0:
            game.completed = True
            await ctx.reply(
                f"No attempts left. The word was {game.word.upper()}. "
                "Use `!wordle` to start a new game."
            )
            return

        game.guesses.append(guess)
        feedback = self._wordle_feedback(game.word, guess)
        board = self._wordle_board(game)

        if guess == game.word:
            game.completed = True
            await ctx.reply(
                f"{guess.upper()} {feedback}\n"
                f"You solved it in {len(game.guesses)} tries! "
                "Use `!wordle` to start a new game."
            )
            return

        if game.attempts_left <= 0:
            game.completed = True
            await ctx.reply(
                f"{guess.upper()} {feedback}\n{board}\n"
                f"Game over. The word was {game.word.upper()}. "
                "Use `!wordle` to start a new game."
            )
            return

        await ctx.reply(
            f"{guess.upper()} {feedback}\n{board}\nAttempts left: {game.attempts_left}"
        )

    async def help_text(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        lines = self._help_lines()
        view = TextPagerView("Backtest Bot Help", lines, page_size=22)
        message = await ctx.reply(view.page_content(), view=view)
        view.message = message

    async def stge_text(self, ctx: commands.Context, *, args: str = "") -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used in a server channel.")
            return
        if not self._is_guild_owner_ctx(ctx):
            await ctx.reply("Only the server owner can change strategies.")
            return
        if not self._is_command_channel(ctx.channel):
            await ctx.reply(
                f"Please use #{self.command_channel_name} for strategy changes."
            )
            return

        tokens = shlex.split(args) if args else []
        if not tokens:
            await ctx.reply("Usage: `!stge set [strategy_name]`")
            return

        if tokens[0].lower() != "set":
            await ctx.reply("Usage: `!stge set [strategy_name]`")
            return

        if len(tokens) > 1:
            name = " ".join(tokens[1:])
            strategies = self._list_strategy_names()
            if name not in strategies:
                await ctx.reply(f"Unknown strategy: {name}")
                return
            self._apply_setting("strategy", "name", name)
            save_settings(self.settings_path, self.settings)
            await ctx.reply(f"Strategy set to {name}.")
            return

        strategies = self._list_strategy_names()
        if not strategies:
            await ctx.reply("No strategies found.")
            return
        notice = "Select a strategy:"
        if len(strategies) > 25:
            notice = (
                "Select a strategy (showing first 25). "
                "Use `!stge set <strategy_name>` to set directly."
            )
            strategies = strategies[:25]
        view = StrategySelectView(self, strategies, ctx.author.id)
        await ctx.reply(notice, view=view)

    @app_commands.command(name="set_config", description="Update config values for backtests.")
    @app_commands.describe(
        section="Config section: data/backtest/strategy",
        key="Key or nested key (use dots for nesting)",
        value="New value",
    )
    async def set_config_slash(
        self,
        interaction: discord.Interaction,
        section: str,
        key: str,
        value: str,
    ) -> None:
        if section not in {"data", "backtest", "strategy"}:
            await interaction.response.send_message(
                "Section must be one of: data, backtest, strategy.", ephemeral=True
            )
            return
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can change settings.", ephemeral=True
            )
            return
        if not self._is_command_channel(interaction.channel):
            await interaction.response.send_message(
                f"Please use #{self.command_channel_name} for config changes.", ephemeral=True
            )
            return

        coerced = self._coerce_value(value)
        self._apply_setting(section, key, coerced)
        save_settings(self.settings_path, self.settings)
        await interaction.response.send_message(
            f"Updated {section}.{key} to {coerced}.", ephemeral=True
        )

    @app_commands.command(
        name="set_config_bulk",
        description="Update multiple config values at once.",
    )
    @app_commands.describe(
        updates="One per line: section.key=value (sections: data/backtest/strategy)",
    )
    async def set_config_bulk_slash(
        self,
        interaction: discord.Interaction,
        updates: str,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can change settings.", ephemeral=True
            )
            return
        if not self._is_command_channel(interaction.channel):
            await interaction.response.send_message(
                f"Please use #{self.command_channel_name} for config changes.", ephemeral=True
            )
            return

        parsed, errors = self._parse_bulk_updates(updates)
        if errors:
            message = "Some lines are invalid:\n" + "\n".join(errors[:8])
            if len(errors) > 8:
                message += f"\n...and {len(errors) - 8} more."
            await interaction.response.send_message(message, ephemeral=True)
            return

        for section, key, value in parsed:
            self._apply_setting(section, key, value)
        save_settings(self.settings_path, self.settings)
        await interaction.response.send_message(
            f"Updated {len(parsed)} setting(s).", ephemeral=True
        )

    @app_commands.command(
        name="config_edit",
        description="Interactive config editor (click to update values).",
    )
    async def config_edit_slash(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message(
                "This command can only be used in a server channel.", ephemeral=True
            )
            return
        if not self._is_guild_owner(interaction):
            await interaction.response.send_message(
                "Only the server owner can change settings.", ephemeral=True
            )
            return
        if not self._is_command_channel(interaction.channel):
            await interaction.response.send_message(
                f"Please use #{self.command_channel_name} for config changes.", ephemeral=True
            )
            return

        entries = self._flatten_settings()
        if not entries:
            await interaction.response.send_message(
                "No editable settings found.", ephemeral=True
            )
            return

        view = ConfigEditorView(self, entries)
        content = "Select a variable to edit.\n" + view._page_summary()
        await interaction.response.send_message(content, view=view, ephemeral=True)
        view.message = await interaction.original_response()


def _get_backtest_bot(ctx: commands.Context) -> "BacktestBot":
    bot = ctx.bot
    if isinstance(bot, BacktestBot):
        return bot
    raise commands.CommandError("Invalid bot instance.")


async def sync_text_command(ctx: commands.Context) -> None:
    await _get_backtest_bot(ctx).sync_text(ctx)


async def download_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).download_text(ctx, args=args)


async def backtest_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).backtest_text(ctx, args=args)


async def entry_diff_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).entry_diff_text(ctx, args=args)


async def grid_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).grid_text(ctx, args=args)


async def wordle_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).wordle_text(ctx, args=args)


async def comp_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).comp_text(ctx, args=args)


async def costom_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).costom_text(ctx, args=args)


async def costom2_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).costom2_text(ctx, args=args)


async def costom3_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).costom3_text(ctx, args=args)


async def costom4_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).costom4_text(ctx, args=args)


async def chart_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).chart_text(ctx, args=args)


async def stge_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).stge_text(ctx, args=args)


async def set_config_text_command(
    ctx: commands.Context,
    section: Optional[str] = None,
    key: Optional[str] = None,
    *,
    value: Optional[str] = None,
) -> None:
    await _get_backtest_bot(ctx).set_config_text(
        ctx,
        section=section,
        key=key,
        value=value,
    )


async def set_config_bulk_text_command(
    ctx: commands.Context,
    *,
    updates: str = "",
) -> None:
    await _get_backtest_bot(ctx).set_config_bulk_text(ctx, updates=updates)


async def config_edit_text_command(ctx: commands.Context) -> None:
    await _get_backtest_bot(ctx).config_edit_text(ctx)


async def delete_data_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).delete_data_text(ctx, args=args)


async def status_text_command(ctx: commands.Context) -> None:
    await _get_backtest_bot(ctx).status_text(ctx)


async def graph_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).graph_text(ctx, args=args)


async def setting_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).setting_text(ctx, args=args)


async def help_text_command(ctx: commands.Context) -> None:
    await _get_backtest_bot(ctx).help_text(ctx)


async def sim_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).sim_text(ctx, args=args)


async def live_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).live_text(ctx, args=args)


async def order_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).order_text(ctx, args=args)


async def close_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).close_text(ctx, args=args)


async def positions_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).positions_text(ctx, args=args)


async def assets_text_command(ctx: commands.Context, *, args: str = "") -> None:
    await _get_backtest_bot(ctx).assets_text(ctx, args=args)
