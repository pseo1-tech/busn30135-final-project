"""
Sector rotation backtest using sentiment signals from the pipeline.

Implements a softmax-weighted sector ETF strategy similar to the
ETF Timing notebook from BUSN 35126, but using sentiment/direction
scores as the signal instead of price momentum.
"""

import os
from io import StringIO

import numpy as np
import pandas as pd

from sector_mapping import SECTOR_ETFS, SECTOR_TO_ETF

# Default path to ETF data parquet
DEFAULT_ETF_DATA_PATH = "/Users/phillipseo/Documents/Booth/Winter 2026/BUSN 35126/Data/ETFData.parquet"

# Fama-French data files (same directory as this file)
_FF5_PATH = os.path.join(os.path.dirname(__file__), "F-F_Research_Data_5_Factors_2x3_daily.csv")
_MOM_PATH = os.path.join(os.path.dirname(__file__), "F-F_Momentum_Factor_daily.csv")


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
    tcost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Run the sector rotation backtest.

    Args:
        sector_signals: DataFrame with [date, sector_etf, avg_sentiment, avg_direction]
        etf_data: Daily ETF return data from the parquet
        signal_dates: List of signal dates (must match dates in sector_signals)
        signal_col: Which column to use as the signal ("avg_direction" or "avg_sentiment")
        b_param: Softmax aggressiveness parameter
        tcost_bps: Round-trip transaction cost in basis points (default 10 bps, academic convention)

    Returns:
        DataFrame with [signal_date, strategy_return, strategy_return_net, equal_weight_return, turnover, tcost, weights...]
    """
    # Compute forward returns between signal dates
    fwd_returns = compute_weekly_returns(etf_data, signal_dates)

    results = []
    # Previous weights start at equal-weight (cost of initial investment is excluded)
    prev_weights = pd.Series(1.0 / len(SECTOR_ETFS), index=SECTOR_ETFS)

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

        # One-way turnover: half the sum of absolute weight changes
        # (buying one sector implies selling another, so total trades = 2x one-way)
        one_way_turnover = float(np.abs(weights.values - prev_weights.values).sum() / 2)
        tcost = one_way_turnover * (tcost_bps / 10_000)
        prev_weights = weights

        # Get forward returns for this period
        period_returns = fwd_returns[fwd_returns["signal_date"] == sig_date]
        ret_vec = pd.Series(dtype=float)
        for etf in SECTOR_ETFS:
            etf_ret = period_returns[period_returns["sector_etf"] == etf]
            ret_vec[etf] = etf_ret["forward_return"].iloc[0] if not etf_ret.empty else 0.0

        # Strategy return: weighted sum (gross), then net of tcosts
        strategy_ret = (weights * ret_vec).sum()
        strategy_ret_net = strategy_ret - tcost

        # Equal-weight benchmark
        equal_ret = ret_vec.mean()

        row = {
            "signal_date": sig_date,
            "strategy_return": strategy_ret,
            "strategy_return_net": strategy_ret_net,
            "equal_weight_return": equal_ret,
            "turnover": round(one_way_turnover, 4),
            "tcost": round(tcost, 6),
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
        results_df["strategy_cumret_net"] = (1 + results_df["strategy_return_net"]).cumprod()
        results_df["equal_weight_cumret"] = (1 + results_df["equal_weight_return"]).cumprod()

    return results_df


def regression_analysis(results_by_signal: dict) -> pd.DataFrame:
    """
    OLS regression of each strategy variant against the equal-weight benchmark.

    Model: r_strategy = alpha + beta * r_ew + epsilon

    Args:
        results_by_signal: {label: results_df} from run_backtest, where each
                           results_df has strategy_return, strategy_return_net,
                           and equal_weight_return columns.

    Returns:
        DataFrame with one row per strategy variant and columns:
        strategy, alpha_ann_pct, beta, r_squared, t_stat_alpha,
        tracking_error_ann_pct, info_ratio, n_periods
    """
    periods_per_year = 52
    rows = []

    for label, df in results_by_signal.items():
        ew = df["equal_weight_return"].values
        n = len(ew)
        X = np.column_stack([np.ones(n), ew])
        XtX_inv = np.linalg.inv(X.T @ X)

        for col, variant in [("strategy_return", "Gross"), ("strategy_return_net", "Net")]:
            if col not in df.columns:
                continue
            y = df[col].values

            # OLS: beta_hat = (X'X)^{-1} X'y
            beta_hat = XtX_inv @ X.T @ y
            alpha_weekly, beta = beta_hat[0], beta_hat[1]

            residuals = y - X @ beta_hat
            rss = np.sum(residuals ** 2)
            tss = np.sum((y - y.mean()) ** 2)
            r_squared = float(1 - rss / tss) if tss > 0 else 0.0

            # Standard error of alpha via OLS covariance matrix
            sigma2 = rss / max(n - 2, 1)
            se_alpha = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
            t_stat = float(alpha_weekly / se_alpha) if se_alpha > 0 else 0.0

            # Annualize
            alpha_ann = float((1 + alpha_weekly) ** periods_per_year - 1)
            tracking_error_ann = float(residuals.std() * np.sqrt(periods_per_year))
            info_ratio = float(alpha_ann / tracking_error_ann) if tracking_error_ann > 0 else 0.0

            rows.append({
                "strategy":                 f"{label} ({variant})",
                "alpha_ann_pct":            round(alpha_ann * 100, 3),
                "beta":                     round(beta, 3),
                "r_squared":                round(r_squared, 3),
                "t_stat_alpha":             round(t_stat, 3),
                "tracking_error_ann_pct":   round(tracking_error_ann * 100, 2),
                "info_ratio":               round(info_ratio, 3),
                "n_periods":                n,
            })

    return pd.DataFrame(rows)


def cross_strategy_regression(results_dfs: dict) -> pd.DataFrame:
    """
    OLS regression of each strategy's gross returns onto every other strategy's gross returns.

    Model: r_A = alpha + beta * r_B + epsilon

    Useful for understanding how much of one signal's return stream is explained by another
    (e.g. Sentiment ~ FinBERT reveals whether LLM sentiment adds anything beyond FinBERT).

    Args:
        results_dfs: {label: results_df} from run_backtest

    Returns:
        DataFrame with columns:
        dependent, independent, alpha_ann_pct, beta, r_squared, t_stat_alpha, n_periods
    """
    periods_per_year = 52
    labels = list(results_dfs.keys())
    rows = []

    for dep_label in labels:
        for ind_label in labels:
            if dep_label == ind_label:
                continue

            dep_df = results_dfs[dep_label]
            ind_df = results_dfs[ind_label]

            # Align on signal_date
            merged = dep_df[["signal_date", "strategy_return"]].merge(
                ind_df[["signal_date", "strategy_return"]].rename(
                    columns={"strategy_return": "ind_return"}
                ),
                on="signal_date",
                how="inner",
            )
            if len(merged) < 3:
                continue

            y = merged["strategy_return"].values
            x = merged["ind_return"].values
            n = len(y)
            X = np.column_stack([np.ones(n), x])
            XtX_inv = np.linalg.inv(X.T @ X)

            beta_hat = XtX_inv @ X.T @ y
            alpha_weekly, beta = beta_hat[0], beta_hat[1]

            residuals = y - X @ beta_hat
            rss = np.sum(residuals ** 2)
            tss = np.sum((y - y.mean()) ** 2)
            r_squared = float(1 - rss / tss) if tss > 0 else 0.0

            sigma2 = rss / max(n - 2, 1)
            se_alpha = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
            t_stat = float(alpha_weekly / se_alpha) if se_alpha > 0 else 0.0

            alpha_ann = float((1 + alpha_weekly) ** periods_per_year - 1)

            rows.append({
                "dependent":     dep_label,
                "independent":   ind_label,
                "alpha_ann_pct": round(alpha_ann * 100, 3),
                "beta":          round(beta, 3),
                "r_squared":     round(r_squared, 3),
                "t_stat_alpha":  round(t_stat, 3),
                "n_periods":     n,
            })

    return pd.DataFrame(rows)


def backtest_summary(results_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the backtest (gross and net of transaction costs)."""
    if results_df.empty:
        return {}

    n_periods = len(results_df)
    periods_per_year = 52  # weekly

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

    summary = {
        "strategy": stats(results_df["strategy_return"], "Strategy (Gross)"),
        "equal_weight": stats(results_df["equal_weight_return"], "Equal Weight"),
    }

    if "strategy_return_net" in results_df.columns:
        summary["strategy_net"] = stats(results_df["strategy_return_net"], "Strategy (Net)")
        summary["avg_turnover"] = round(results_df["turnover"].mean() * 100, 2)
        summary["avg_tcost_bps"] = round(results_df["tcost"].mean() * 10_000, 2)

    return summary


def _read_ff_csv(path: str) -> pd.DataFrame:
    """
    Parse a Ken French CSV file, skipping the text preamble and Copyright footer.

    The header row starts with a comma (date column has no name); the following
    line starts with a digit (YYYYMMDD date). Returns a DataFrame with a 'date'
    column (datetime) and factor columns in decimal form (divided by 100).
    """
    with open(path) as f:
        lines = f.readlines()

    # Locate the header row: starts with ',' and is followed by a data row
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(","):
            # Verify next non-empty line starts with a digit
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                if candidate and candidate[0].isdigit():
                    header_idx = i
                    break
        if header_idx is not None:
            break

    if header_idx is None:
        raise ValueError(f"Could not locate header row in {path}")

    # Trim Copyright footer
    end_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if "Copyright" in lines[i]:
            end_idx = i
            break

    df = pd.read_csv(StringIO("".join(lines[header_idx:end_idx])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.index.name = "date"
    # Clean column names: strip whitespace, replace hyphens with underscores
    df.columns = [c.strip().replace("-", "_") for c in df.columns]
    # Replace sentinel missing values and convert percent → decimal
    df = df.replace(-99.99, np.nan).replace(-999.0, np.nan)
    df = df / 100.0
    return df.reset_index()


def load_ff_factors(ff5_path: str = None, mom_path: str = None) -> pd.DataFrame:
    """
    Load Fama-French 5 Factors + Momentum (daily) into a single DataFrame.

    Returns DataFrame with columns:
        date, Mkt_RF, SMB, HML, RMW, CMA, RF, MOM
    All values in decimal (not percent). Returns empty DataFrame if files not found.
    """
    ff5_path = ff5_path or _FF5_PATH
    mom_path = mom_path or _MOM_PATH

    if not os.path.exists(ff5_path) or not os.path.exists(mom_path):
        return pd.DataFrame()

    ff5 = _read_ff_csv(ff5_path)
    mom = _read_ff_csv(mom_path).rename(columns={"Mom": "MOM"})

    factors = ff5.merge(mom, on="date", how="left")
    return factors.sort_values("date").reset_index(drop=True)


def factor_regression(
    results_dfs: dict,
    ff_factors: pd.DataFrame,
    signal_dates: list = None,
) -> pd.DataFrame:
    """
    Fama-French 6-factor regression for each strategy.

    Model: r_strategy - RF = α + β₁·Mkt_RF + β₂·SMB + β₃·HML
                               + β₄·RMW + β₅·CMA + β₆·MOM + ε

    Factor returns are compounded over each holding period to match the
    backtest's weekly return horizon.

    Args:
        results_dfs:   {label: results_df} from run_backtest
        ff_factors:    DataFrame from load_ff_factors()
        signal_dates:  Full list of signal dates (including terminal date) so the
                       last holding period's factor returns can be computed.
                       If omitted the last period is dropped.

    Returns:
        DataFrame with one row per strategy and columns:
        strategy, alpha_ann_pct, t_stat_alpha, r_squared,
        beta_Mkt_RF, beta_SMB, beta_HML, beta_RMW, beta_CMA, beta_MOM, n_periods
    """
    if ff_factors.empty:
        return pd.DataFrame()

    periods_per_year = 52
    factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    available_factors = [fc for fc in factor_cols if fc in ff_factors.columns]

    # Build the full sorted list of period boundaries once (shared across strategies)
    if signal_dates is not None:
        all_dates = sorted([pd.Timestamp(d) for d in signal_dates])
    else:
        all_dates = None  # will be inferred per-strategy below

    rows = []

    for label, df in results_dfs.items():
        # Use provided full date list; fall back to dates in results_df (misses last period)
        dates = all_dates if all_dates is not None else sorted(df["signal_date"].tolist())

        # Aggregate factor returns over each holding period (t_signal, t_next]
        period_factors = []
        for i in range(len(dates) - 1):
            t0 = pd.Timestamp(dates[i])
            t1 = pd.Timestamp(dates[i + 1])
            mask = (ff_factors["date"] > t0) & (ff_factors["date"] <= t1)
            period_ff = ff_factors[mask]

            row = {"signal_date": t0}
            if period_ff.empty:
                row["RF_period"] = 0.0
                for fc in available_factors:
                    row[fc] = 0.0
            else:
                row["RF_period"] = float((1 + period_ff["RF"]).prod() - 1)
                for fc in available_factors:
                    col_data = period_ff[fc].dropna()
                    row[fc] = float((1 + col_data).prod() - 1) if not col_data.empty else 0.0

            period_factors.append(row)

        if not period_factors:
            continue

        factors_df = pd.DataFrame(period_factors)
        factors_df["signal_date"] = pd.to_datetime(factors_df["signal_date"])

        merged = df[["signal_date", "strategy_return"]].merge(
            factors_df, on="signal_date", how="inner"
        )

        if len(merged) < len(available_factors) + 2:
            continue

        # Excess strategy return (strategy return minus period risk-free rate)
        y = (merged["strategy_return"] - merged["RF_period"]).values
        n = len(y)

        X_factors = merged[available_factors].values
        X = np.column_stack([np.ones(n), X_factors])
        XtX_inv = np.linalg.inv(X.T @ X)

        beta_hat = XtX_inv @ X.T @ y
        alpha_weekly = beta_hat[0]

        residuals = y - X @ beta_hat
        rss = np.sum(residuals ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        r_squared = float(1 - rss / tss) if tss > 0 else 0.0

        sigma2 = rss / max(n - len(beta_hat), 1)
        se_alpha = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
        t_stat = float(alpha_weekly / se_alpha) if se_alpha > 0 else 0.0

        alpha_ann = float((1 + alpha_weekly) ** periods_per_year - 1)

        out = {
            "strategy":        label,
            "alpha_ann_pct":   round(alpha_ann * 100, 3),
            "t_stat_alpha":    round(t_stat, 3),
            "r_squared":       round(r_squared, 3),
            "n_periods":       n,
        }
        for fc, b in zip(available_factors, beta_hat[1:]):
            out[f"beta_{fc}"] = round(float(b), 3)

        rows.append(out)

    return pd.DataFrame(rows)
