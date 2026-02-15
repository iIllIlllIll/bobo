from __future__ import annotations

import re


_MINUTE_RULE = re.compile(r"^(?P<value>\d+)m$")


def normalize_pandas_rule(rule: str) -> str:
    """Normalize common interval strings to pandas-compatible offset aliases."""
    if not isinstance(rule, str):
        raise TypeError("rule must be a string")

    cleaned = rule.strip()
    match = _MINUTE_RULE.match(cleaned)
    if match:
        return f"{match.group('value')}min"

    return cleaned
