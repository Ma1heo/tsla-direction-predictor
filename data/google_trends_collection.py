"""
Google Trends Collection — Granular, Categorized Search Terms
=============================================================
Collects weekly Google Trends data for actionable Tesla-related
search terms, categorized into signal groups:

  - risk:     terms that spike during negative events
  - investor: terms reflecting active trading/investment interest
  - product:  terms tied to product launches and catalysts
  - brand:    baseline brand/ticker awareness (anchor term)

Uses overlapping 5-year windows to get weekly granularity over
the full date range, then normalizes across windows.

Usage:
    python data/google_trends_collection.py

Requires:
    pip install pytrends
"""

import time
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "data" else SCRIPT_DIR
DATA_RAW = PROJECT_ROOT / "data" / "raw"

OUTPUT_CSV = DATA_RAW / "google_trends_granular.csv"

# Anchor term included in every batch for cross-normalization
ANCHOR = "Tesla"

# Categorized search terms
# pytrends max 5 per request, anchor takes 1 slot → 4 terms per batch
SEARCH_TERMS = {
    # Risk / negative signals
    "risk": [
        "tesla recall",
        "tesla crash",
        "tesla lawsuit",
        "tesla investigation",
    ],
    # Investor attention
    "investor": [
        "buy tesla stock",
        "sell tesla stock",
        "tsla earnings",
        "tesla stock price",
    ],
    # Product catalysts
    "product": [
        "tesla fsd",
        "tesla robotaxi",
        "cybertruck delivery",
        "tesla model 2",
    ],
}

# Time windows: 5-year chunks give weekly data, with 6-month overlap
WINDOWS = [
    ("2016-01-01", "2021-06-30"),
    ("2021-01-01", "2026-02-28"),
]

PAUSE_BETWEEN_REQUESTS = 8  # seconds, conservative to avoid 429s


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def fetch_batch(pytrends: TrendReq, keywords: list[str],
                timeframe: str, retries: int = 3) -> pd.DataFrame | None:
    """Fetch interest over time for a batch of keywords."""
    for attempt in range(retries):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo="")
            df = pytrends.interest_over_time()
            if not df.empty and "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            return df
        except Exception as e:
            wait = PAUSE_BETWEEN_REQUESTS * (attempt + 1)
            print(f"    Retry {attempt+1}/{retries} after error: {e}")
            print(f"    Waiting {wait}s...")
            time.sleep(wait)
    return None


def normalize_windows(dfs: list[pd.DataFrame], anchor_col: str) -> pd.DataFrame:
    """Merge overlapping time windows using the anchor term to normalize scale.

    The anchor term appears in all windows. In the overlap period, we compute
    the ratio between windows and scale the earlier window to match the later one.
    """
    if len(dfs) == 1:
        return dfs[0]

    # Start with the most recent window (highest fidelity)
    result = dfs[-1].copy()

    for i in range(len(dfs) - 2, -1, -1):
        earlier = dfs[i].copy()

        # Find overlap period
        overlap_start = max(earlier.index.min(), result.index.min())
        overlap_end = min(earlier.index.max(), result.index.max())

        overlap_earlier = earlier.loc[overlap_start:overlap_end, anchor_col]
        overlap_later = result.loc[overlap_start:overlap_end, anchor_col]

        # Compute scaling factor from anchor in overlap
        if len(overlap_earlier) > 0 and overlap_earlier.mean() > 0:
            scale = overlap_later.mean() / overlap_earlier.mean()
        else:
            scale = 1.0

        print(f"  Window {i}: scaling factor = {scale:.3f} "
              f"(overlap {overlap_start.date()} to {overlap_end.date()}, "
              f"{len(overlap_earlier)} points)")

        # Scale earlier window's non-anchor columns
        for col in earlier.columns:
            if col != anchor_col:
                earlier[col] = earlier[col] * scale

        # Also scale the anchor for consistency
        earlier[anchor_col] = earlier[anchor_col] * scale

        # Prepend the non-overlapping portion of the earlier window
        pre_overlap = earlier.loc[:overlap_start].iloc[:-1]  # exclude first overlap date
        result = pd.concat([pre_overlap, result])

    return result.sort_index()


def collect_category(pytrends: TrendReq, category: str,
                     terms: list[str]) -> pd.DataFrame | None:
    """Collect terms in a category WITHOUT the anchor to get proper resolution.

    When niche terms like 'tesla recall' are queried alongside 'Tesla',
    they get squished to 0 because Tesla dominates the 0-100 scale.
    Instead, query category terms together so they're scaled relative
    to each other.
    """
    print(f"\n  Category: {category} ({len(terms)} terms)")

    # Use the first term in the category as the intra-category anchor
    # for cross-window normalization
    cat_anchor = terms[0]

    window_dfs = []
    for w_start, w_end in WINDOWS:
        timeframe = f"{w_start} {w_end}"
        print(f"    Window {w_start} to {w_end}: {terms}")

        df = fetch_batch(pytrends, terms, timeframe)
        if df is None:
            print(f"    FAILED — skipping window")
            continue

        window_dfs.append(df)
        print(f"    Got {len(df)} weekly points, "
              f"date range {df.index.min().date()} to {df.index.max().date()}")
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not window_dfs:
        print(f"    No data collected for {category}")
        return None

    # Normalize across windows using first term as anchor
    merged = normalize_windows(window_dfs, cat_anchor)

    # Rename columns with category prefix
    rename = {}
    for col in merged.columns:
        clean = col.lower().replace(" ", "_").replace("-", "_")
        rename[col] = f"gtrend_{category}_{clean}"
    merged = merged.rename(columns=rename)

    return merged


def main():
    print("=" * 60)
    print("GOOGLE TRENDS — GRANULAR COLLECTION")
    print("=" * 60)

    pytrends = TrendReq(hl="en-US", tz=360)

    all_dfs = []

    # First, collect the anchor term alone for baseline brand awareness
    print("\n  Collecting anchor term: Tesla")
    anchor_windows = []
    for w_start, w_end in WINDOWS:
        timeframe = f"{w_start} {w_end}"
        df = fetch_batch(pytrends, [ANCHOR], timeframe)
        if df is not None:
            anchor_windows.append(df)
            print(f"    Window {w_start}-{w_end}: {len(df)} points")
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if anchor_windows:
        anchor_df = normalize_windows(anchor_windows, ANCHOR)
        anchor_df = anchor_df.rename(columns={ANCHOR: "gtrend_tesla"})
        all_dfs.append(anchor_df)

    # Then collect each category (without anchor, for proper resolution)
    for category, terms in SEARCH_TERMS.items():
        df = collect_category(pytrends, category, terms)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data collected. Exiting.")
        return

    # Merge all DataFrames on date index
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = result.join(df, how="outer")

    # Sort and clean
    result = result.sort_index()
    result.index.name = "Date"

    # Fill any small gaps (holidays etc) with forward fill
    result = result.ffill()

    # Save
    result.to_csv(OUTPUT_CSV)
    print(f"\n{'=' * 60}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Shape: {result.shape}")
    print(f"Date range: {result.index.min().date()} to {result.index.max().date()}")
    print(f"Columns: {list(result.columns)}")
    print(f"\nSample (last 5 rows):")
    print(result.tail().to_string())

    # Category summaries
    print(f"\nCategory signal strength (mean values):")
    for cat in SEARCH_TERMS:
        cat_cols = [c for c in result.columns if f"gtrend_{cat}_" in c]
        if cat_cols:
            means = result[cat_cols].mean()
            print(f"  {cat}: {dict(means.round(1))}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
