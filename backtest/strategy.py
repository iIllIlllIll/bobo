from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of -1/0/1 signals aligned to data rows."""
        raise NotImplementedError

    def signal_reasons(self, data: pd.DataFrame) -> pd.Series:
        """Optional: return a Series of entry-reason strings aligned to data rows."""
        return pd.Series([""] * len(data), index=data.index)

    def exit_signals(self, data: pd.DataFrame) -> pd.Series:
        """Optional: return a Series of -1/0/1 exit signals aligned to data rows."""
        return pd.Series([0] * len(data), index=data.index)
