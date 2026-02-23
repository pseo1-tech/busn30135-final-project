"""
Sector rotation backtest using sentiment signals from the pipeline.

Implements a softmax-weighted sector ETF strategy similar to the
ETF Timing notebook from BUSN 35126, but using sentiment/direction
scores as the signal instead of price momentum.
"""

import numpy as np
import pandas as pd

from sector_mapping import SECTOR_ETFS, SECTOR_TO_ETF

# Default path to ETF data parquet
DEFAULT_ETF_DATA_PATH = "/Users/phillipseo/Documents/Booth/Winter 2026/BUSN 35126/Data/ETFData.parquet"


def load_etf_data(parquet_path: str = DEFAULT_ETF_DATA_PATH) -> pd.DataFrame:
    """Load the ETF parquet and return only sector ETF daily data."""
    df = pd.read_parquet(parquet_path)
    df = df[df["ticker"].isin(SECTOR_ETFS)][["date", "ticker", "retd", "retM"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def get_trading_dates(etf_data: pd.DataFrame) -> pd.Series:
    """Get sorted unique trading dates from the ETF data."""
    return etf_data["date"].drop_duplicates().sort_values().reset_index(drop=True)


def sample_weekly_dates(trading_dates: pd.Series, start_date: str, end_date: str) -> list[str]:
    """
    Sample weekly signal dates from actual trading dates in the parquet.
    Takes the last trading day of each week (typically Friday).

    Returns list of date strings (YYYY-MM-DD) that exist in the parquet.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Filter to date range
    mask = (trading_dates >= start) & (trading_dates <= end)
    dates_in_range = trading_dates[mask]

    if dates_in_range.empty:
        return []

    # Group by week, take last trading day per week
    weekly = dates_in_range.groupby(dates_in_range.dt.to_period("W")).max()

    return [d.strftime("%Y-%m-%d") for d in weekly]


def sample_daily_dates(trading_dates: pd.Series, start_date: str, end_date: str) -> list[str]:
    """Return all trading dates in the range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (trading_dates >= start) & (trading_dates <= end)
    return [d.strftime("%Y-%m-%d") for d in trading_dates[mask]]


def compute_weekly_returns(etf_data: pd.DataFrame, signal_dates: list[str]) -> pd.DataFrame:
    """
    Compute forward returns between consecutive signal dates for each sector ETF.

    For each signal date t, the forward return is the cumulative daily return
    from t+1 to t+1_next (the next signal date). This is the return earned
    by holding the portfolio from signal date to the next rebalance.

    Returns DataFrame: [signal_date, sector_etf, forward_return]
    """
    signal_ts = sorted([pd.Timestamp(d) for d in signal_dates])
    all_dates = etf_data["date"].drop_duplicates().sort_values().reset_index(drop=True)

    rows = []
    for i in range(len(signal_ts) - 1):
        t_signal = signal_ts[i]
        t_next = signal_ts[i + 1]

        # Forward returns: from day after signal to next signal date (inclusive)
        mask = (etf_data["date"] > t_signal) & (etf_data["date"] <= t_next)
        period_data = etf_data[mask]

        for etf in SECTOR_ETFS:
            etf_rows = period_data[period_data["ticker"] == etf]
            if etf_rows.empty:
                cum_ret = 0.0
            else:
                # Cumulative return: product of (1 + daily ret) - 1
                cum_ret = (1 + etf_rows["retd"]).prod() - 1

            rows.append({
                "signal_date": t_signal.strftime("%Y-%m-%d"),
                "sector_etf": etf,
                "forward_return": cum_ret,
            })

    return pd.DataFrame(rows)


def softmax_weights(signals: pd.Series, b_param: float = 5.0) -> pd.Series:
    """
    Compute softmax portfolio weights from signals.

    w_i = exp(b * signal_i) / sum_j exp(b * signal_j)

    Args:
        signals: Series of signal values per sector ETF
        b_param: Aggressiveness parameter. Higher = more concentrated. Default 5.
    """
    exp_vals = np.exp(b_param * signals)
    return exp_vals / exp_vals.sum()


def run_backtest(
    sector_signals: pd.DataFrame,
    etf_data: pd.DataFrame,
    signal_dates: list[str],
    signal_col: str = "avg_direction",
    b_param: float = 5.0,
) -> pd.DataFrame:
    """
    Run the sector rotation backtest.

    Args:
        sector_signals: DataFrame with [date, sector_etf, avg_sentiment, avg_direction]
        etf_data: Daily ETF return data from the parquet
        signal_dates: List of signal dates (must match dates in sector_signals)
        signal_col: Which column to use as the signal ("avg_direction" or "avg_sentiment")
        b_param: Softmax aggressiveness parameter

    Returns:
        DataFrame with [signal_date, strategy_return, equal_weight_return, weights...]
    """
    # Compute forward returns between signal dates
    fwd_returns = compute_weekly_returns(etf_data, signal_dates)

    results = []

    for i, sig_date in enumerate(signal_dates[:-1]):
        # Get signals for this date
        date_signals = sector_signals[sector_signals["date"] == pd.Timestamp(sig_date)]

        if date_signals.empty:
            # No signal for this date -> equal weight
            weights = pd.Series(1.0 / len(SECTOR_ETFS), index=SECTOR_ETFS)
        else:
            # Build signal vector (fill missing sectors with neutral)
            signal_vec = pd.Series(dtype=float)
            for etf in SECTOR_ETFS:
                etf_row = date_signals[date_signals["sector_etf"] == etf]
                if not etf_row.empty:
                    signal_vec[etf] = etf_row[signal_col].iloc[0]
                else:
                    # Neutral default: 0 for sentiment, 5.5 for direction
                    neutral = 0.0 if signal_col == "avg_sentiment" else 5.5
                    signal_vec[etf] = neutral

            # Normalize direction scores to center around 0 for softmax
            if signal_col == "avg_direction":
                signal_vec = (signal_vec - 5.5) / 4.5  # maps [1,10] to ~[-1, +1]

            weights = softmax_weights(signal_vec, b_param=b_param)

        # Get forward returns for this period
        period_returns = fwd_returns[fwd_returns["signal_date"] == sig_date]
        ret_vec = pd.Series(dtype=float)
        for etf in SECTOR_ETFS:
            etf_ret = period_returns[period_returns["sector_etf"] == etf]
            ret_vec[etf] = etf_ret["forward_return"].iloc[0] if not etf_ret.empty else 0.0

        # Strategy return: weighted sum
        strategy_ret = (weights * ret_vec).sum()

        # Equal-weight benchmark
        equal_ret = ret_vec.mean()

        row = {
            "signal_date": sig_date,
            "strategy_return": strategy_ret,
            "equal_weight_return": equal_ret,
        }
        # Store weights for analysis
        for etf in SECTOR_ETFS:
            row[f"w_{etf}"] = weights.get(etf, 1.0 / len(SECTOR_ETFS))

        results.append(row)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df["signal_date"] = pd.to_datetime(results_df["signal_date"])

        # Cumulative returns
        results_df["strategy_cumret"] = (1 + results_df["strategy_return"]).cumprod()
        results_df["equal_weight_cumret"] = (1 + results_df["equal_weight_return"]).cumprod()

    return results_df


def backtest_summary(results_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the backtest."""
    if results_df.empty:
        return {}

    n_periods = len(results_df)
    periods_per_year = 52  # weekly

    strat = results_df["strategy_return"]
    ew = results_df["equal_weight_return"]

    def stats(returns, label):
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + returns).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        return {
            "label": label,
            "total_return": round(total_ret * 100, 2),
            "annualized_return": round(ann_ret * 100, 2),
            "annualized_vol": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "n_periods": n_periods,
        }

    return {
        "strategy": stats(strat, "Sentiment Strategy"),
        "equal_weight": stats(ew, "Equal Weight"),
    }
