from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests


class DiscordNotifier:
    def __init__(self, bot_token: str, channel_id: str) -> None:
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.base_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"

    def send_message(self, content: str, chart_paths: Optional[List[Path]] = None) -> None:
        headers = {
            "Authorization": f"Bot {self.bot_token}",
        }
        payload: Dict[str, Any] = {"content": content}
        if chart_paths:
            file_handles = []
            try:
                files = {}
                for idx, path in enumerate(chart_paths):
                    handle = Path(path).open("rb")
                    file_handles.append(handle)
                    mime_type, _ = mimetypes.guess_type(str(path))
                    if not mime_type:
                        mime_type = "application/octet-stream"
                    files[f"files[{idx}]"] = (Path(path).name, handle, mime_type)
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data={"payload_json": json.dumps(payload)},
                    files=files,
                    timeout=20,
                )
            finally:
                for handle in file_handles:
                    handle.close()
        else:
            headers["Content-Type"] = "application/json"
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10,
            )
        response.raise_for_status()
