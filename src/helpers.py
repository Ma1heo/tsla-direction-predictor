"""
Reusable helper functions for the TSLA direction predictor project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
FIGURES = PROJECT_ROOT / "outputs" / "figures"


def align_to_trading_days(df: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a dataframe to trading days only, forward-filling gaps.

    Handles source data on non-trading dates (e.g. weekly Sunday dates,
    month-start dates) by combining both index sets before forward-filling,
    then filtering to trading days only.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    # Combine source dates + trading dates so ffill propagates from source → trading days
    combined = df.index.union(trading_dates).sort_values()
    df = df.reindex(combined).ffill()
    # Keep only trading days
    df = df.reindex(trading_dates)
    return df


def create_target(df: pd.DataFrame, close_col: str = "Close") -> pd.Series:
    """target = 1 if next day's close > today's close, else 0."""
    return (df[close_col].shift(-1) > df[close_col]).astype(int)


def plot_save(fig, name: str):
    """Save a matplotlib figure to the figures directory."""
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    return pd.DataFrame({"missing": total, "pct": pct}).query("missing > 0").sort_values("pct", ascending=False)
