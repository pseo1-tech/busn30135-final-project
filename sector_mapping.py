"""
Ticker-to-sector mapping using a static CSV seed.

Maps individual stock tickers (e.g. AAPL) to GICS sectors (e.g. Information Technology)
and then to sector ETFs (e.g. XLK). Unknown tickers are silently dropped — no network
calls are made at runtime.
"""

import os
import pandas as pd

# GICS sector name -> sector ETF ticker
SECTOR_TO_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

SECTOR_ETFS = list(SECTOR_TO_ETF.values())

# Path to the seed CSV (same directory as this file)
_SEED_CSV = os.path.join(os.path.dirname(__file__), "sector_map.csv")

# In-memory cache: ticker -> gics_sector
_cache: dict[str, str] = {}
_cache_loaded = False


def _load_cache():
    """Load the seed CSV into the in-memory cache."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    if os.path.exists(_SEED_CSV):
        df = pd.read_csv(_SEED_CSV)
        for _, row in df.iterrows():
            _cache[row["ticker"].upper()] = row["gics_sector"]
    _cache_loaded = True


def get_sector(ticker: str) -> str | None:
    """
    Get the GICS sector for a ticker from the seed CSV.
    Returns None if the ticker is not in the seed.
    """
    _load_cache()
    return _cache.get(ticker.upper())


def get_sector_etf(ticker: str) -> str | None:
    """
    Get the sector ETF for a stock ticker.
    e.g. AAPL -> XLK, JPM -> XLF
    Returns None if the ticker is not in the seed CSV.
    """
    sector = get_sector(ticker)
    if sector:
        return SECTOR_TO_ETF.get(sector)
    return None


def map_scores_to_sectors(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a scores DataFrame (date, ticker, avg_sentiment, avg_direction)
    and add sector_etf column. Drops rows where sector can't be determined.

    Returns DataFrame with columns: date, ticker, sector_etf, avg_sentiment, avg_direction
    """
    _load_cache()

    sector_etfs = [get_sector_etf(t) for t in scores_df["ticker"]]

    scores_df = scores_df.copy()
    scores_df["sector_etf"] = sector_etfs

    # Drop unmappable tickers
    n_before = len(scores_df)
    scores_df = scores_df.dropna(subset=["sector_etf"]).reset_index(drop=True)
    n_dropped = n_before - len(scores_df)

    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{n_before} rows with unmappable tickers")

    return scores_df


def aggregate_sector_signals(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-ticker scores into per-sector signals.

    Input: DataFrame with columns [date, ticker, sector_etf, avg_sentiment, avg_direction, num_articles]
    Output: DataFrame with columns [date, sector_etf, avg_sentiment, avg_direction, num_articles, num_tickers]

    Weighted by num_articles (tickers with more articles get more weight).
    """
    if scores_df.empty:
        return pd.DataFrame()

    def weighted_avg(group, col):
        weights = group["num_articles"]
        if weights.sum() == 0:
            return group[col].mean()
        return (group[col] * weights).sum() / weights.sum()

    has_finbert = "avg_finbert_sentiment" in scores_df.columns

    rows = []
    for (date, etf), group in scores_df.groupby(["date", "sector_etf"]):
        row = {
            "date": date,
            "sector_etf": etf,
            "avg_sentiment": round(weighted_avg(group, "avg_sentiment"), 4),
            "avg_direction": round(weighted_avg(group, "avg_direction"), 2),
            "num_articles": int(group["num_articles"].sum()),
            "num_tickers": len(group),
        }
        if has_finbert:
            valid = group["avg_finbert_sentiment"].notna()
            if valid.any():
                g = group[valid]
                w = g["num_articles"]
                row["avg_finbert_sentiment"] = round(
                    (g["avg_finbert_sentiment"] * w).sum() / w.sum(), 4
                )
            else:
                row["avg_finbert_sentiment"] = None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["date", "sector_etf"]).reset_index(drop=True)
