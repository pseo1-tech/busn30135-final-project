import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import json
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no window popup
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import agents
from graph import app
from models import SentimentState
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from ui import console


def display_results(state: SentimentState, label: str = ""):
    """Display per-article scores and per-ticker aggregates as Rich tables."""
    title_prefix = f"[dim]{label}[/dim] " if label else ""

    # --- Per-Article Table ---
    art_table = Table(title=f"{title_prefix}[bold cyan]Per-Article Scores[/bold cyan]", show_header=True, header_style="bold magenta")
    art_table.add_column("Title", style="cyan", max_width=45)
    art_table.add_column("Tickers", style="yellow", width=12)
    art_table.add_column("Sentiment", justify="right", style="green", width=10)
    art_table.add_column("Direction", justify="right", style="blue", width=10)
    art_table.add_column("Reasoning", style="dim", max_width=50)

    for score_dict in state.article_scores:
        sent = score_dict["sentiment_score"]
        direction = score_dict["price_direction_score"]
        sent_color = "green" if sent > 0 else ("red" if sent < 0 else "white")
        dir_color = "green" if direction > 5 else ("red" if direction < 5 else "white")

        art_table.add_row(
            score_dict["title"][:45],
            ", ".join(score_dict.get("tickers", []))[:12],
            f"[{sent_color}]{sent:+.2f}[/{sent_color}]",
            f"[{dir_color}]{direction}/10[/{dir_color}]",
            score_dict.get("reasoning", "")[:50],
        )

    console.print("\n")
    console.print(art_table)

    # --- Per-Ticker Aggregate Table ---
    if state.ticker_aggregates:
        agg_table = Table(title=f"\n{title_prefix}[bold cyan]Ticker Aggregates[/bold cyan]", show_header=True, header_style="bold magenta")
        agg_table.add_column("Ticker", style="bold cyan", width=10)
        agg_table.add_column("Articles", justify="right", style="yellow", width=9)
        agg_table.add_column("Avg Sentiment", justify="right", style="green", width=14)
        agg_table.add_column("Avg Direction", justify="right", style="blue", width=14)
        agg_table.add_column("Avg FinBERT", justify="right", style="magenta", width=13)

        for agg in state.ticker_aggregates:
            sent_color = "green" if agg.avg_sentiment > 0 else ("red" if agg.avg_sentiment < 0 else "white")
            dir_color = "green" if agg.avg_price_direction > 5 else ("red" if agg.avg_price_direction < 5 else "white")
            fb = agg.avg_finbert_sentiment
            fb_color = "green" if fb is not None and fb > 0 else ("red" if fb is not None and fb < 0 else "white")
            fb_str = f"[{fb_color}]{fb:+.3f}[/{fb_color}]" if fb is not None else "[dim]N/A[/dim]"

            agg_table.add_row(
                agg.ticker,
                str(agg.num_articles),
                f"[{sent_color}]{agg.avg_sentiment:+.3f}[/{sent_color}]",
                f"[{dir_color}]{agg.avg_price_direction:.1f}/10[/{dir_color}]",
                fb_str,
            )

        console.print(agg_table)
        console.print("\n")


def display_usage(state: SentimentState):
    """Display usage summary."""
    total = state.total_usage
    table = Table(title="[bold cyan]Usage Summary[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("Node", style="cyan", width=15)
    table.add_column("Calls", justify="right", style="yellow")
    table.add_column("Input Tokens", justify="right", style="green")
    table.add_column("Output Tokens", justify="right", style="green")
    table.add_column("Cost", justify="right", style="red")

    for node_name, node_agg in sorted(state.node_usage.items()):
        table.add_row(
            node_name,
            str(node_agg['num_calls']),
            f"{node_agg['input_tokens']:,}",
            f"{node_agg['output_tokens']:,}",
            f"${node_agg['cost']:.4f}",
        )

    table.add_row(
        "[bold]TOTAL[/bold]", "[bold]-[/bold]",
        f"[bold]{total['input_tokens']:,}[/bold]",
        f"[bold]{total['output_tokens']:,}[/bold]",
        f"[bold]${total['total_cost']:.4f}[/bold]",
        style="bold white on black",
    )
    console.print(table)


def display_combined_summary(all_day_results: list):
    """Display a combined summary table across all dates in the range."""
    table = Table(title="\n[bold cyan]Date Range Summary[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("Date", style="cyan", width=12)
    table.add_column("Articles", justify="right", style="yellow", width=9)
    table.add_column("Avg Sentiment", justify="right", style="green", width=14)
    table.add_column("Avg Direction", justify="right", style="blue", width=14)
    table.add_column("Cost", justify="right", style="red", width=10)

    total_articles = 0
    total_cost = 0.0

    for day in all_day_results:
        n = day["num_articles"]
        total_articles += n
        cost = day["usage"]["total_cost"]
        total_cost += cost

        if n == 0:
            table.add_row(day["date"], "0", "[dim]n/a[/dim]", "[dim]n/a[/dim]", f"${cost:.4f}")
            continue

        avg_s = sum(s["sentiment_score"] for s in day["article_scores"]) / n
        avg_d = sum(s["price_direction_score"] for s in day["article_scores"]) / n
        s_color = "green" if avg_s > 0 else ("red" if avg_s < 0 else "white")
        d_color = "green" if avg_d > 5 else ("red" if avg_d < 5 else "white")

        table.add_row(
            day["date"],
            str(n),
            f"[{s_color}]{avg_s:+.3f}[/{s_color}]",
            f"[{d_color}]{avg_d:.1f}/10[/{d_color}]",
            f"${cost:.4f}",
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_articles}[/bold]",
        "", "",
        f"[bold]${total_cost:.4f}[/bold]",
        style="bold white on black",
    )
    console.print(table)


def build_scores_dataframe(all_day_results: list) -> pd.DataFrame:
    """
    Build a flat DataFrame from pipeline results with one row per (date, ticker) pair.
    Pulls from ticker_aggregates (already computed by aggregator_node) to avoid
    re-aggregating and to include avg_finbert_sentiment.

    Columns: date, ticker, num_articles, avg_sentiment, avg_direction, avg_finbert_sentiment
    """
    rows = []
    for day in all_day_results:
        date_str = day["date"]
        for agg in day.get("ticker_aggregates", []):
            rows.append({
                "date": date_str,
                "ticker": agg["ticker"],
                "num_articles": agg["num_articles"],
                "avg_sentiment": agg["avg_sentiment"],
                "avg_direction": agg["avg_price_direction"],
                "avg_finbert_sentiment": agg.get("avg_finbert_sentiment"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def update_usage_tracking(final_state_dict, node_name):
    """Same usage tracking as lab_01."""
    from agents import aggregate_node_usage
    all_calls = final_state_dict.get("llm_calls", [])
    node_agg = aggregate_node_usage(all_calls, node_name)

    if "node_usage" not in final_state_dict:
        final_state_dict["node_usage"] = {}
    final_state_dict["node_usage"][node_name] = node_agg

    # Recompute totals from scratch to avoid double-counting
    final_state_dict["total_usage"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
    for n_agg in final_state_dict["node_usage"].values():
        final_state_dict["total_usage"]["input_tokens"] += n_agg["input_tokens"]
        final_state_dict["total_usage"]["output_tokens"] += n_agg["output_tokens"]
        final_state_dict["total_usage"]["total_tokens"] += n_agg["total_tokens"]
        final_state_dict["total_usage"]["total_cost"] += n_agg["cost"]


async def run_pipeline(date_str: str, ticker: str = None, logs_dir: str = None, cached_prompt: str = None):
    """Run the sentiment analysis pipeline for a single date.

    Args:
        cached_prompt: Pre-generated optimized analyst prompt to reuse (skips prompt_optimizer).
    """
    initial_state = SentimentState(
        date_str=date_str,
        ticker_filter=ticker,
        logs_dir=logs_dir,
        optimized_analyst_prompt=cached_prompt,
    )

    console.print(Panel(
        f"[bold blue]News Sentiment Agent[/bold blue]\nDate: {date_str}" + (f"\nTicker: {ticker}" if ticker else ""),
        title="Starting Pipeline",
    ))

    final_state_dict = initial_state.model_dump()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Initializing...", total=None)

        # Keys that use operator.add in the state model — must be accumulated, not replaced
        additive_keys = {"article_scores", "llm_calls"}

        async for event in app.astream(initial_state, config={"recursion_limit": 50}):
            for node_name, state_update in event.items():
                if state_update is None:
                    continue

                for key, value in state_update.items():
                    if key in additive_keys and isinstance(value, list):
                        final_state_dict.setdefault(key, []).extend(value)
                    else:
                        final_state_dict[key] = value

                update_usage_tracking(final_state_dict, node_name)
                progress.update(task_id, description=f"Agent working: [bold]{node_name}[/bold]")

                if node_name == "news_collector":
                    num = len(state_update.get("collected_articles", []))
                    console.print(Panel(f"Fetched {num} articles", title="[bold cyan]News Collector[/bold cyan]", border_style="cyan"))
                elif node_name == "analyst":
                    scores = state_update.get("article_scores", [])
                    for s in scores:
                        console.print(f"  [dim]Scored: {s['title'][:50]} -> Sent: {s['sentiment_score']:+.2f}, Dir: {s['price_direction_score']}/10[/dim]")

                await asyncio.sleep(0.1)

    console.print("[bold]Pipeline Complete.[/bold]\n")

    final_state = SentimentState(**final_state_dict)
    return final_state


def display_backtest_results(summary: dict, results_df: pd.DataFrame):
    """Display backtest summary as a Rich table."""
    if not summary:
        console.print("[yellow]No backtest results to display.[/yellow]")
        return

    strat = summary["strategy"]
    strat_net = summary.get("strategy_net")
    ew = summary["equal_weight"]

    table = Table(title="\n[bold cyan]Backtest Results[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Strategy (Gross)", justify="right", style="green", width=18)
    if strat_net:
        table.add_column("Strategy (Net)", justify="right", style="yellow", width=16)
    table.add_column("Equal Weight", justify="right", style="blue", width=14)

    metric_keys = [
        ("Total Return",      "total_return",      "{:+.2f}%"),
        ("Annualized Return", "annualized_return",  "{:+.2f}%"),
        ("Annualized Vol",    "annualized_vol",     "{:.2f}%"),
        ("Sharpe Ratio",      "sharpe_ratio",       "{:.3f}"),
        ("Max Drawdown",      "max_drawdown",       "{:.2f}%"),
        ("Periods",           "n_periods",          "{}"),
    ]

    for label, key, fmt in metric_keys:
        s_val = fmt.format(strat[key])
        e_val = fmt.format(ew[key])
        if strat_net:
            n_val = fmt.format(strat_net[key])
            table.add_row(label, s_val, n_val, e_val)
        else:
            table.add_row(label, s_val, e_val)

    console.print(table)

    # Turnover / tcost footnote
    if strat_net and "avg_turnover" in summary:
        console.print(
            f"  [dim]Avg one-way turnover: {summary['avg_turnover']:.1f}%  |  "
            f"Avg tcost drag: {summary['avg_tcost_bps']:.2f} bps/period[/dim]"
        )

    # Excess return lines
    gross_excess = strat["total_return"] - ew["total_return"]
    color = "green" if gross_excess > 0 else "red"
    console.print(f"\n  [{color}]Strategy excess return (gross): {gross_excess:+.2f}%[/{color}]")
    if strat_net:
        net_excess = strat_net["total_return"] - ew["total_return"]
        color_net = "green" if net_excess > 0 else "red"
        console.print(f"  [{color_net}]Strategy excess return (net):   {net_excess:+.2f}%[/{color_net}]")


_BACKTEST_SIGNALS = [
    ("avg_direction",         "Direction"),
    ("avg_sentiment",         "Sentiment"),
    ("avg_finbert_sentiment",  "FinBERT"),
]


def display_multi_backtest_results(all_summaries: dict):
    """Display a scoreboard table comparing all signal strategies side by side."""
    if not all_summaries:
        console.print("[yellow]No backtest results to display.[/yellow]")
        return

    table = Table(
        title="\n[bold cyan]Backtest Comparison — All Signals[/bold cyan]",
        show_header=True, header_style="bold magenta"
    )
    table.add_column("Strategy", style="cyan", width=22)
    table.add_column("Total Return", justify="right", width=13)
    table.add_column("Ann. Return",  justify="right", width=12)
    table.add_column("Ann. Vol",     justify="right", width=10)
    table.add_column("Sharpe",       justify="right", width=8)
    table.add_column("Max DD",       justify="right", width=10)

    def color_val(v, fmt, higher_is_better=True):
        is_good = v > 0 if higher_is_better else v < 0
        color = "green" if is_good else ("red" if (v < 0 if higher_is_better else v > 0) else "white")
        return f"[{color}]{fmt.format(v)}[/{color}]"

    ew = None
    for i, (label, summary) in enumerate(all_summaries.items()):
        if ew is None:
            ew = summary["equal_weight"]

        for key, row_label, style in [
            ("strategy",     f"{label} (Gross)", "green"),
            ("strategy_net", f"{label} (Net)",   "yellow"),
        ]:
            s = summary.get(key)
            if s is None:
                continue
            table.add_row(
                f"[{style}]{row_label}[/{style}]",
                color_val(s["total_return"],      "{:+.2f}%"),
                color_val(s["annualized_return"],  "{:+.2f}%"),
                f"{s['annualized_vol']:.2f}%",
                color_val(s["sharpe_ratio"],       "{:.3f}"),
                color_val(s["max_drawdown"],       "{:.2f}%", higher_is_better=False),
            )

        # Turnover footnote per signal
        if "avg_turnover" in summary:
            console.print(
                f"  [dim]{label}: avg one-way turnover {summary['avg_turnover']:.1f}%  "
                f"→  avg tcost drag {summary['avg_tcost_bps']:.2f} bps/period[/dim]"
            ) if i == 0 else None  # printed after table below

        if i < len(all_summaries) - 1:
            table.add_section()

    # Equal weight benchmark row
    if ew:
        table.add_section()
        table.add_row(
            "[blue]Equal Weight[/blue]",
            color_val(ew["total_return"],      "{:+.2f}%"),
            color_val(ew["annualized_return"],  "{:+.2f}%"),
            f"{ew['annualized_vol']:.2f}%",
            color_val(ew["sharpe_ratio"],       "{:.3f}"),
            color_val(ew["max_drawdown"],       "{:.2f}%", higher_is_better=False),
        )

    console.print(table)

    # Turnover footnotes for all signals
    for label, summary in all_summaries.items():
        if "avg_turnover" in summary:
            console.print(
                f"  [dim]{label}: avg one-way turnover {summary['avg_turnover']:.1f}%  "
                f"→  avg tcost drag {summary['avg_tcost_bps']:.2f} bps/period[/dim]"
            )

    # Excess return vs equal weight
    console.print()
    for label, summary in all_summaries.items():
        ew_ret = summary["equal_weight"]["total_return"]
        for key, variant in [("strategy", "gross"), ("strategy_net", "net")]:
            s = summary.get(key)
            if s is None:
                continue
            excess = s["total_return"] - ew_ret
            color = "green" if excess > 0 else "red"
            console.print(f"  [{color}]{label} ({variant}) excess return: {excess:+.2f}%[/{color}]")


def run_all_backtests(
    sector_signals: pd.DataFrame,
    etf_data,
    signal_dates: list,
    b_param: float,
    tcost_bps: float,
) -> tuple[dict, dict]:
    """
    Run backtest for every available signal column.
    Returns (summaries, results_dfs) where both are {label: ...} dicts.
    """
    from backtest import run_backtest, backtest_summary as bt_summary

    summaries = {}
    results_dfs = {}
    for col, label in _BACKTEST_SIGNALS:
        if col not in sector_signals.columns or sector_signals[col].isna().all():
            continue
        df = run_backtest(
            sector_signals=sector_signals,
            etf_data=etf_data,
            signal_dates=signal_dates,
            signal_col=col,
            b_param=b_param,
            tcost_bps=tcost_bps,
        )
        if not df.empty:
            summaries[label] = bt_summary(df)
            results_dfs[label] = df

    return summaries, results_dfs


def display_regression_results(reg_df: pd.DataFrame):
    """Display OLS regression results as a Rich table."""
    if reg_df.empty:
        return

    table = Table(
        title="\n[bold cyan]Regression vs. Equal-Weight Benchmark[/bold cyan]  "
              "[dim](r_strategy = α + β·r_ew + ε)[/dim]",
        show_header=True, header_style="bold magenta"
    )
    table.add_column("Strategy",          style="cyan",  width=22)
    table.add_column("Alpha (ann.)",      justify="right", width=13)
    table.add_column("Beta",              justify="right", width=7)
    table.add_column("R²",               justify="right", width=7)
    table.add_column("t(α)",             justify="right", width=8)
    table.add_column("Track. Err.",       justify="right", width=12)
    table.add_column("Info Ratio",        justify="right", width=11)

    for _, row in reg_df.iterrows():
        alpha = row["alpha_ann_pct"]
        t = row["t_stat_alpha"]
        ir = row["info_ratio"]
        alpha_color = "green" if alpha > 0 else "red"
        t_color     = "green" if t > 2 else ("yellow" if t > 1 else "red")
        ir_color    = "green" if ir > 0 else "red"

        table.add_row(
            row["strategy"],
            f"[{alpha_color}]{alpha:+.3f}%[/{alpha_color}]",
            f"{row['beta']:.3f}",
            f"{row['r_squared']:.3f}",
            f"[{t_color}]{t:.3f}[/{t_color}]",
            f"{row['tracking_error_ann_pct']:.2f}%",
            f"[{ir_color}]{ir:.3f}[/{ir_color}]",
        )

    console.print(table)
    console.print(
        "  [dim]α: annualized intercept (outperformance unexplained by benchmark exposure)  "
        "|  t(α) > 2 ≈ significant at 95%  |  IR = α / tracking error[/dim]"
    )


def display_cross_regression_results(cross_df: pd.DataFrame):
    """Display cross-strategy OLS regression results as a Rich table."""
    if cross_df.empty:
        return

    table = Table(
        title="\n[bold cyan]Cross-Strategy Regression[/bold cyan]  "
              "[dim](r_A = α + β·r_B + ε)[/dim]",
        show_header=True, header_style="bold magenta"
    )
    table.add_column("Dependent (A)",   style="cyan",  width=18)
    table.add_column("Independent (B)", style="yellow", width=18)
    table.add_column("Alpha (ann.)",    justify="right", width=13)
    table.add_column("Beta",            justify="right", width=7)
    table.add_column("R²",             justify="right", width=7)
    table.add_column("t(α)",           justify="right", width=8)

    for _, row in cross_df.iterrows():
        alpha = row["alpha_ann_pct"]
        t = row["t_stat_alpha"]
        alpha_color = "green" if alpha > 0 else "red"
        t_color = "green" if t > 2 else ("yellow" if t > 1 else "red")

        table.add_row(
            row["dependent"],
            row["independent"],
            f"[{alpha_color}]{alpha:+.3f}%[/{alpha_color}]",
            f"{row['beta']:.3f}",
            f"{row['r_squared']:.3f}",
            f"[{t_color}]{t:.3f}[/{t_color}]",
        )

    console.print(table)
    console.print(
        "  [dim]α: incremental return of A not explained by B's return stream  "
        "|  t(α) > 2 ≈ significant at 95%[/dim]"
    )


def display_factor_regression_results(factor_df: pd.DataFrame):
    """Display Fama-French 6-factor regression results as a Rich table."""
    if factor_df.empty:
        return

    # Determine which beta columns are present
    beta_cols = [c for c in factor_df.columns if c.startswith("beta_")]
    factor_labels = {
        "beta_Mkt_RF": "Mkt-RF",
        "beta_SMB":    "SMB",
        "beta_HML":    "HML",
        "beta_RMW":    "RMW",
        "beta_CMA":    "CMA",
        "beta_MOM":    "MOM",
    }

    table = Table(
        title="\n[bold cyan]Fama-French 6-Factor Regression[/bold cyan]  "
              "[dim](r_strategy − RF = α + Σ βᵢ·Fᵢ + ε)[/dim]",
        show_header=True, header_style="bold magenta"
    )
    table.add_column("Strategy",    style="cyan", width=18)
    table.add_column("Alpha (ann.)", justify="right", width=13)
    table.add_column("t(α)",        justify="right", width=8)
    table.add_column("R²",         justify="right", width=7)
    for bc in beta_cols:
        table.add_column(factor_labels.get(bc, bc.replace("beta_", "")), justify="right", width=8)

    for _, row in factor_df.iterrows():
        alpha = row["alpha_ann_pct"]
        t = row["t_stat_alpha"]
        alpha_color = "green" if alpha > 0 else "red"
        t_color = "green" if t > 2 else ("yellow" if t > 1 else "red")

        cells = [
            row["strategy"],
            f"[{alpha_color}]{alpha:+.3f}%[/{alpha_color}]",
            f"[{t_color}]{t:.3f}[/{t_color}]",
            f"{row['r_squared']:.3f}",
        ]
        for bc in beta_cols:
            cells.append(f"{row[bc]:.3f}")

        table.add_row(*cells)

    console.print(table)
    console.print(
        "  [dim]α: annualized return not explained by the 6 Fama-French factors  "
        "|  t(α) > 2 ≈ significant at 95%[/dim]"
    )


def display_b_param_results(tune_results: dict):
    """Display b_param sweep (in-sample) and out-of-sample performance."""
    if not tune_results:
        return

    signal_col = tune_results["signal_col"]
    best_b     = tune_results["best_b"]
    is_start, is_end   = tune_results["is_dates"]
    oos_start, oos_end = tune_results["oos_dates"]
    sweep      = tune_results["sweep"]
    oos        = tune_results["oos_summary"]

    # --- In-sample sweep table ---
    sweep_table = Table(
        title=f"\n[bold cyan]b_param Sweep — {signal_col} (In-Sample: {is_start} to {is_end})[/bold cyan]",
        show_header=True, header_style="bold magenta",
    )
    sweep_table.add_column("b_param",      justify="right", width=9)
    sweep_table.add_column("IS Sharpe",    justify="right", width=11)
    sweep_table.add_column("IS Ann Ret",   justify="right", width=11)
    sweep_table.add_column("IS Total Ret", justify="right", width=12)
    sweep_table.add_column("IS Max DD",    justify="right", width=10)

    for _, row in sweep.iterrows():
        is_best = row["b_param"] == best_b
        style   = "bold green" if is_best else ""
        marker  = " ◀ best" if is_best else ""
        sweep_table.add_row(
            f"[{style}]{row['b_param']}{marker}[/{style}]" if style else f"{row['b_param']}{marker}",
            f"[{style}]{row['is_sharpe']:+.3f}[/{style}]" if style else f"{row['is_sharpe']:+.3f}",
            f"[{style}]{row['is_ann_ret_pct']:+.2f}%[/{style}]" if style else f"{row['is_ann_ret_pct']:+.2f}%",
            f"[{style}]{row['is_total_ret_pct']:+.2f}%[/{style}]" if style else f"{row['is_total_ret_pct']:+.2f}%",
            f"[{style}]{row['is_max_dd_pct']:+.2f}%[/{style}]" if style else f"{row['is_max_dd_pct']:+.2f}%",
        )
    console.print(sweep_table)

    # --- Out-of-sample results ---
    if not oos:
        return

    oos_table = Table(
        title=f"[bold cyan]Out-of-Sample Performance — b={best_b} (OOS: {oos_start} to {oos_end})[/bold cyan]",
        show_header=True, header_style="bold magenta",
    )
    oos_table.add_column("Strategy",    style="cyan",    width=22)
    oos_table.add_column("Ann Ret",     justify="right", width=10)
    oos_table.add_column("Sharpe",      justify="right", width=8)
    oos_table.add_column("Max DD",      justify="right", width=10)
    oos_table.add_column("Total Ret",   justify="right", width=11)

    def _row(label, s, style=""):
        c = lambda v, good: f"[green]{v}[/green]" if good else f"[red]{v}[/red]"
        oos_table.add_row(
            f"[{style}]{label}[/{style}]" if style else label,
            c(f"{s['annualized_return']:+.2f}%", s["annualized_return"] > 0),
            c(f"{s['sharpe_ratio']:+.3f}",       s["sharpe_ratio"] > 0),
            c(f"{s['max_drawdown']:+.2f}%",      s["max_drawdown"] > -10),
            c(f"{s['total_return']:+.2f}%",      s["total_return"] > 0),
        )

    _row(f"Strategy Gross (b={best_b})", oos["strategy"],     "green")
    if "strategy_net" in oos:
        _row(f"Strategy Net   (b={best_b})", oos["strategy_net"], "yellow")
    _row("Equal Weight",                  oos["equal_weight"])
    console.print(oos_table)


def plot_sector_weights(results_dfs: dict, save_dir: str = None, split_dates: dict = None):
    """
    Plot sector ETF weights over time for each strategy as stacked area charts.

    One subplot per strategy, x-axis = signal date, y-axis = portfolio weight (0–1).
    If split_dates = {label: date_str} is provided, draws a vertical dashed line
    and shades/labels the In-Sample and Out-of-Sample regions.
    Saves to save_dir/sector_weights.png if provided.
    """
    from sector_mapping import SECTOR_ETFS, SECTOR_TO_ETF

    ETF_TO_SECTOR = {v: k for k, v in SECTOR_TO_ETF.items()}

    if not results_dfs:
        return

    n = len(results_dfs)
    colors = plt.cm.tab20.colors[:len(SECTOR_ETFS)]

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle("Sector ETF Weights Over Time", fontsize=14, fontweight="bold", y=1.01)

    for row_idx, (label, df) in enumerate(results_dfs.items()):
        ax = axes[row_idx, 0]

        weight_cols = [f"w_{etf}" for etf in SECTOR_ETFS if f"w_{etf}" in df.columns]
        sector_names = [ETF_TO_SECTOR.get(c.replace("w_", ""), c.replace("w_", "")) for c in weight_cols]
        dates = df["signal_date"].values
        weights = df[weight_cols].values.T

        ax.stackplot(dates, weights, labels=sector_names, colors=colors, alpha=0.85)

        eq = 1.0 / len(SECTOR_ETFS)
        ax.axhline(eq, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
                   label=f"Equal weight ({eq:.2f})")

        # IS/OOS split line and region labels
        if split_dates and label in split_dates:
            split_dt = pd.Timestamp(split_dates[label])
            ax.axvline(split_dt, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.text(split_dt, 1.02, "▼ OOS", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="black")
            x_min = pd.Timestamp(dates[0])
            x_max = pd.Timestamp(dates[-1])
            mid_is  = x_min + (split_dt - x_min) / 2
            mid_oos = split_dt + (x_max - split_dt) / 2
            ax.text(mid_is,  0.97, "In-Sample",     transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="dimgray", style="italic")
            ax.text(mid_oos, 0.97, "Out-of-Sample", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="dimgray", style="italic")

        ax.set_title(f"{label} Strategy", fontsize=11, fontweight="bold")
        ax.set_ylabel("Portfolio Weight")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=8, framealpha=0.8, ncol=1)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "sector_weights.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"[green]Saved sector weights plot to: {path}[/green]")
    else:
        plt.show()


def plot_cumulative_returns(tune_results: dict, save_dir: str = None):
    """
    Plot cumulative returns for each strategy in two panels side-by-side:
      Left : Total-space  — gross, net, and equal-weight cumulative return
      Right: Active-space — gross and net cumulative active return vs equal weight

    One row per signal type. IS/OOS boundary marked with a vertical dotted line.
    """
    if not tune_results:
        return

    labels = [label for label, tr in tune_results.items() if tr and
              not tr.get("is_results", pd.DataFrame()).empty]
    if not labels:
        return

    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle("Cumulative Returns — Total & Active Space",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_idx, label in enumerate(labels):
        tr      = tune_results[label]
        best_b  = tr["best_b"]
        df      = pd.concat([tr["is_results"], tr["oos_results"]], ignore_index=True)
        split_dt = pd.Timestamp(tr["oos_dates_list"][0])

        dates   = pd.to_datetime(df["signal_date"])
        gross   = (1 + df["strategy_return"]).cumprod() - 1
        net     = (1 + df["strategy_return_net"]).cumprod() - 1
        ew      = (1 + df["equal_weight_return"]).cumprod() - 1
        act_gross = (1 + df["strategy_return"]     - df["equal_weight_return"]).cumprod() - 1
        act_net   = (1 + df["strategy_return_net"] - df["equal_weight_return"]).cumprod() - 1

        def _add_split(ax):
            ax.axvline(split_dt, color="black", linestyle=":", linewidth=1.5, alpha=0.6)
            ymin, ymax = ax.get_ylim()
            mid_is  = dates.iloc[0]  + (split_dt - dates.iloc[0])  / 2
            mid_oos = split_dt       + (dates.iloc[-1] - split_dt) / 2
            ax.text(mid_is,  ymax, "In-Sample",     ha="center", va="top",
                    fontsize=7, color="dimgray", style="italic")
            ax.text(mid_oos, ymax, "Out-of-Sample", ha="center", va="top",
                    fontsize=7, color="dimgray", style="italic")

        # --- Total space ---
        ax_total = axes[row_idx, 0]
        ax_total.plot(dates, gross * 100, label="Gross",        color="steelblue",  linewidth=1.8)
        ax_total.plot(dates, net   * 100, label="Net",          color="darkorange", linewidth=1.8)
        ax_total.plot(dates, ew    * 100, label="Equal Weight", color="gray",
                      linewidth=1.2, linestyle="--")
        ax_total.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax_total.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax_total.set_title(f"{label} (b={best_b}) — Total", fontsize=10, fontweight="bold")
        ax_total.set_ylabel("Cumulative Return")
        ax_total.tick_params(axis="x", rotation=30)
        ax_total.legend(fontsize=8)
        _add_split(ax_total)

        # --- Active space ---
        ax_active = axes[row_idx, 1]
        ax_active.plot(dates, act_gross * 100, label="Gross Active", color="steelblue",  linewidth=1.8)
        ax_active.plot(dates, act_net   * 100, label="Net Active",   color="darkorange", linewidth=1.8)
        ax_active.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
                          label="Equal Weight (0%)")
        ax_active.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax_active.set_title(f"{label} (b={best_b}) — Active vs Equal Weight",
                            fontsize=10, fontweight="bold")
        ax_active.set_ylabel("Cumulative Active Return")
        ax_active.tick_params(axis="x", rotation=30)
        ax_active.legend(fontsize=8)
        _add_split(ax_active)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "cumulative_returns.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"[green]Saved cumulative returns plot to: {path}[/green]")
    else:
        plt.show()


def generate_date_range(start_date: str, end_date: str) -> list:
    """Generate a list of date strings from start_date to end_date (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if end < start:
        raise ValueError(f"End date ({end_date}) is before start date ({start_date})")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


async def run_single_date(date_str: str, ticker: str, base_logs_dir: str, cached_prompt: str, semaphore: asyncio.Semaphore):
    """Run the pipeline for one date, gated by a concurrency semaphore."""
    async with semaphore:
        day_logs = f"{base_logs_dir}/{date_str}" if base_logs_dir else None
        if day_logs:
            os.makedirs(day_logs, exist_ok=True)

        # Resume: if this date already completed successfully, load and return cached result
        cached_result_path = f"{day_logs}/results.json" if day_logs else None
        if cached_result_path and os.path.exists(cached_result_path):
            try:
                with open(cached_result_path) as f:
                    cached = json.load(f)
                console.print(f"  [dim]⏭  {date_str}: skipping (cached)[/dim]")
                return cached
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt cache — fall through and re-run

        state = await run_pipeline(date_str, ticker=ticker, logs_dir=day_logs, cached_prompt=cached_prompt)

        day_result = {
            "date": date_str,
            "num_articles": len(state.article_scores),
            "article_scores": state.article_scores,
            "ticker_aggregates": [a.model_dump() for a in state.ticker_aggregates],
            "usage": state.total_usage,
        }

        # Save per-date results
        if day_logs:
            with open(f"{day_logs}/results.json", "w") as f:
                json.dump(day_result, f, indent=2)

        return day_result


async def run_date_range(
    dates: list,
    ticker: str = None,
    base_logs_dir: str = None,
    concurrency: int = 5,
    run_backtest_flag: bool = False,
    b_param: float = 5.0,
    tcost_bps: float = 10.0,
    etf_data_path: str = None,
):
    """Run the pipeline over a range of dates in parallel, reusing the optimized prompt across all days."""

    console.print(Panel(
        f"[bold blue]Date Range Mode (parallel)[/bold blue]\n"
        f"Range: {dates[0]} to {dates[-1]} ({len(dates)} signal dates)\n"
        f"Concurrency: {concurrency} dates at a time"
        + (f"\nTicker: {ticker}" if ticker else ""),
        title="Multi-Day Pipeline",
    ))

    # Step 1: Generate optimized prompt once before parallelizing
    console.print("[bold yellow]Step 1/2: Generating optimized analyst prompt...[/bold yellow]")
    from agents import run_prompt_optimizer_standalone
    cached_prompt = run_prompt_optimizer_standalone()

    # Step 2: Run all dates in parallel with bounded concurrency
    console.print(f"\n[bold yellow]Step 2/2: Scoring {len(dates)} dates (up to {concurrency} in parallel)...[/bold yellow]")
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        run_single_date(date_str, ticker, base_logs_dir, cached_prompt, semaphore)
        for date_str in dates
    ]
    all_day_results = await asyncio.gather(*tasks)

    # Sort results by date (gather preserves order, but be explicit)
    all_day_results = sorted(all_day_results, key=lambda d: d["date"])

    # Display per-date results
    for day in all_day_results:
        if day["num_articles"] > 0:
            console.print(f"\n[dim]{day['date']}: {day['num_articles']} articles scored[/dim]")

    # Display combined summary
    display_combined_summary(all_day_results)

    # Build and save DataFrame (date x ticker)
    scores_df = build_scores_dataframe(all_day_results)
    if base_logs_dir and not scores_df.empty:
        csv_path = f"{base_logs_dir}/scores.csv"
        scores_df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved {len(scores_df)} rows to: {csv_path}[/green]")

    # Run backtest if requested
    if run_backtest_flag and not scores_df.empty:
        console.print("\n[bold cyan]Running sector rotation backtest...[/bold cyan]")
        from backtest import load_etf_data
        from sector_mapping import aggregate_sector_signals, map_scores_to_sectors

        # Map ticker scores to sectors
        mapped = map_scores_to_sectors(scores_df)
        sector_signals = aggregate_sector_signals(mapped)

        if sector_signals.empty:
            console.print("[yellow]No sector signals could be computed. Skipping backtest.[/yellow]")
        else:
            console.print(f"  Sector signals: {len(sector_signals)} (date, sector) pairs")

            # Save sector signals
            if base_logs_dir:
                sector_path = f"{base_logs_dir}/sector_signals.csv"
                sector_signals.to_csv(sector_path, index=False)
                console.print(f"[green]Saved sector signals to: {sector_path}[/green]")

            # Load ETF return data
            etf_data = load_etf_data(etf_data_path) if etf_data_path else load_etf_data()

            # Run backtest for all available signals
            all_summaries, results_dfs = run_all_backtests(
                sector_signals=sector_signals,
                etf_data=etf_data,
                signal_dates=dates,
                b_param=b_param,
                tcost_bps=tcost_bps,
            )

            if all_summaries:
                display_multi_backtest_results(all_summaries)

                from backtest import (
                    regression_analysis, cross_strategy_regression,
                    factor_regression, load_ff_factors, tune_b_param,
                )

                # Step 1: b_param sweep — IS tuning + OOS evaluation
                tune_results = {}
                for signal_col, label in _BACKTEST_SIGNALS:
                    tune_res = tune_b_param(
                        sector_signals, etf_data, dates,
                        signal_col=signal_col, tcost_bps=tcost_bps,
                    )
                    tune_results[label] = tune_res
                    display_b_param_results(tune_res)

                # Step 2: Regressions on OOS results
                oos_results_dfs = {
                    label: tr["oos_results"]
                    for label, tr in tune_results.items()
                    if tr and not tr.get("oos_results", pd.DataFrame()).empty
                }
                oos_signal_dates = next(
                    (tr["oos_dates_list"] for tr in tune_results.values() if tr and "oos_dates_list" in tr),
                    dates,
                )
                if oos_results_dfs:
                    ff_factors = load_ff_factors()
                    reg_df   = regression_analysis(oos_results_dfs)
                    cross_df = cross_strategy_regression(oos_results_dfs)
                    factor_df = factor_regression(oos_results_dfs, ff_factors, signal_dates=oos_signal_dates)
                    display_regression_results(reg_df)
                    display_cross_regression_results(cross_df)
                    display_factor_regression_results(factor_df)

                    if base_logs_dir:
                        reg_df.to_csv(f"{base_logs_dir}/regression_results.csv", index=False)
                        console.print(f"[green]Saved regression results to: {base_logs_dir}/regression_results.csv[/green]")
                        cross_df.to_csv(f"{base_logs_dir}/cross_strategy_regression.csv", index=False)
                        console.print(f"[green]Saved cross-strategy regression to: {base_logs_dir}/cross_strategy_regression.csv[/green]")
                        if not factor_df.empty:
                            factor_df.to_csv(f"{base_logs_dir}/factor_regression.csv", index=False)
                            console.print(f"[green]Saved FF6 factor regression to: {base_logs_dir}/factor_regression.csv[/green]")

                # Step 3: Sector weights plot — IS+OOS concatenated with split line
                combined_dfs = {}
                split_dates  = {}
                for label, tr in tune_results.items():
                    if tr and not tr.get("is_results", pd.DataFrame()).empty and not tr.get("oos_results", pd.DataFrame()).empty:
                        combined_dfs[label] = pd.concat([tr["is_results"], tr["oos_results"]], ignore_index=True)
                        split_dates[label]  = tr["oos_dates_list"][0]
                plot_sector_weights(combined_dfs or results_dfs, save_dir=base_logs_dir,
                                    split_dates=split_dates or None)
                plot_cumulative_returns(tune_results, save_dir=base_logs_dir)
            else:
                console.print("[yellow]Backtest produced no results (need at least 2 signal dates).[/yellow]")

    return all_day_results


if __name__ == "__main__":
    import argparse

    # Try to load secrets from course_utils if available
    try:
        from course_utils import env_setup
        env_setup.init()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="News Sentiment & Price Direction Scoring Agent")
    parser.add_argument("--date", default=None, help="Single date to query (e.g. 2024-06-15)")
    parser.add_argument("--start-date", default=None, help="Start date for range (e.g. 2024-06-01)")
    parser.add_argument("--end-date", default=None, help="End date for range (e.g. 2024-06-30)")
    parser.add_argument("--freq", default="weekly", choices=["weekly", "daily"],
                        help="Signal frequency for date range mode (default: weekly)")
    parser.add_argument("--ticker", default=None, help="Optional ticker to filter articles (e.g. TSLA)")
    parser.add_argument("--model", default=None, help="Model spec (provider:model), e.g. together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    parser.add_argument("--api-key", default=None, help="Massive API key (or set MASSIVE_API_KEY env var)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max parallel dates in range mode (default: 5)")
    parser.add_argument("--backtest", action="store_true", help="Run sector rotation backtest after scoring")
    parser.add_argument("--resume", default=None, metavar="LOGS_DIR",
                        help="Resume an interrupted range run by reusing an existing logs directory (skips already-completed dates)")
    parser.add_argument("--from-signals", default=None, metavar="CSV",
                        help="Skip pipeline entirely and run backtest directly from a saved sector_signals.csv")
    parser.add_argument("--signal", default="avg_direction", choices=["avg_direction", "avg_sentiment", "avg_finbert_sentiment"],
                        help="Signal column for backtest (default: avg_direction)")
    parser.add_argument("--b-param", type=float, default=5.0, help="Softmax aggressiveness parameter (default: 5.0)")
    parser.add_argument("--tcost", type=float, default=10.0, help="Round-trip transaction cost in bps (default: 10 bps)")
    parser.add_argument("--etf-data", default=None, help="Path to ETFData.parquet (auto-detected if not set)")

    args = parser.parse_args()

    # --- Standalone backtest mode: load cached sector_signals.csv, skip pipeline ---
    if args.from_signals:
        import sys
        from backtest import load_etf_data

        console.print(f"[bold cyan]Standalone backtest mode — loading signals from:[/bold cyan] {args.from_signals}")
        sector_signals = pd.read_csv(args.from_signals, parse_dates=["date"])
        if sector_signals.empty:
            console.print("[bold red]Error:[/bold red] sector_signals.csv is empty.")
            sys.exit(1)

        signal_dates = sorted(sector_signals["date"].dt.strftime("%Y-%m-%d").unique().tolist())
        console.print(f"  {len(signal_dates)} signal dates: {signal_dates[0]} to {signal_dates[-1]}")

        etf_data = load_etf_data(args.etf_data) if args.etf_data else load_etf_data()
        all_summaries, results_dfs = run_all_backtests(
            sector_signals=sector_signals,
            etf_data=etf_data,
            signal_dates=signal_dates,
            b_param=args.b_param,
            tcost_bps=args.tcost,
        )

        if not all_summaries:
            console.print("[yellow]Backtest produced no results (need at least 2 signal dates).[/yellow]")
            sys.exit(0)

        display_multi_backtest_results(all_summaries)

        from backtest import (
            regression_analysis, cross_strategy_regression,
            factor_regression, load_ff_factors, tune_b_param,
        )

        # Step 1: b_param sweep — IS tuning + OOS evaluation
        tune_results = {}
        for signal_col, label in _BACKTEST_SIGNALS:
            tune_res = tune_b_param(
                sector_signals, etf_data, signal_dates,
                signal_col=signal_col, tcost_bps=args.tcost,
            )
            tune_results[label] = tune_res
            display_b_param_results(tune_res)

        # Step 2: Regressions on OOS results
        signals_dir = os.path.dirname(args.from_signals)
        oos_results_dfs = {
            label: tr["oos_results"]
            for label, tr in tune_results.items()
            if tr and not tr.get("oos_results", pd.DataFrame()).empty
        }
        oos_signal_dates = next(
            (tr["oos_dates_list"] for tr in tune_results.values() if tr and "oos_dates_list" in tr),
            signal_dates,
        )
        if oos_results_dfs:
            ff_factors = load_ff_factors()
            reg_df    = regression_analysis(oos_results_dfs)
            cross_df  = cross_strategy_regression(oos_results_dfs)
            factor_df = factor_regression(oos_results_dfs, ff_factors, signal_dates=oos_signal_dates)
            display_regression_results(reg_df)
            display_cross_regression_results(cross_df)
            display_factor_regression_results(factor_df)

            reg_df.to_csv(os.path.join(signals_dir, "regression_results.csv"), index=False)
            console.print(f"[green]Saved regression results to: {signals_dir}/regression_results.csv[/green]")
            cross_df.to_csv(os.path.join(signals_dir, "cross_strategy_regression.csv"), index=False)
            console.print(f"[green]Saved cross-strategy regression to: {signals_dir}/cross_strategy_regression.csv[/green]")
            if not factor_df.empty:
                factor_df.to_csv(os.path.join(signals_dir, "factor_regression.csv"), index=False)
                console.print(f"[green]Saved FF6 factor regression to: {signals_dir}/factor_regression.csv[/green]")

        # Step 3: Sector weights plot — IS+OOS concatenated with split line
        combined_dfs = {}
        split_dates  = {}
        for label, tr in tune_results.items():
            if tr and not tr.get("is_results", pd.DataFrame()).empty and not tr.get("oos_results", pd.DataFrame()).empty:
                combined_dfs[label] = pd.concat([tr["is_results"], tr["oos_results"]], ignore_index=True)
                split_dates[label]  = tr["oos_dates_list"][0]
        plot_sector_weights(combined_dfs or results_dfs, save_dir=signals_dir,
                            split_dates=split_dates or None)
        plot_cumulative_returns(tune_results, save_dir=signals_dir)
        sys.exit(0)

    # Validate date arguments
    if args.date and (args.start_date or args.end_date):
        console.print("[bold red]Error:[/bold red] Use either --date (single) OR --start-date/--end-date (range), not both.")
        import sys; sys.exit(1)
    if not args.date and not (args.start_date and args.end_date):
        console.print("[bold red]Error:[/bold red] Provide --date for a single day, or --start-date and --end-date for a range.")
        import sys; sys.exit(1)
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        console.print("[bold red]Error:[/bold red] Both --start-date and --end-date are required for range mode.")
        import sys; sys.exit(1)

    # Set massive API key if provided
    if args.api_key:
        os.environ["MASSIVE_API_KEY"] = args.api_key

    # Configure LLM
    try:
        with console.status("[bold green]Verifying model access...[/bold green]", spinner="dots"):
            agents.configure_agents(model_spec=args.model)
    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        import sys; sys.exit(1)

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.date:
        # Single date mode (original behavior)
        slug = f"{args.date}_{args.ticker or 'all'}"
        logs_dir = f"logs/{timestamp}_{slug}"
        os.makedirs(logs_dir, exist_ok=True)

        async def _single():
            state = await run_pipeline(args.date, ticker=args.ticker, logs_dir=logs_dir)
            display_results(state)
            display_usage(state)

            # Build and save DataFrame
            day_result = {
                "date": args.date,
                "num_articles": len(state.article_scores),
                "article_scores": state.article_scores,
                "usage": state.total_usage,
            }
            df = build_scores_dataframe([day_result])
            csv_path = f"{logs_dir}/scores.csv"
            df.to_csv(csv_path, index=False)
            console.print(f"[green]Saved {len(df)} rows to: {csv_path}[/green]")

        asyncio.run(_single())
    else:
        # Date range mode — sample dates from ETF parquet trading calendar
        from backtest import load_etf_data, get_trading_dates, sample_weekly_dates, sample_daily_dates

        console.print(f"[bold]Loading ETF trading calendar for {args.freq} date sampling...[/bold]")
        etf_data = load_etf_data(args.etf_data) if args.etf_data else load_etf_data()
        trading_dates = get_trading_dates(etf_data)

        if args.freq == "weekly":
            dates = sample_weekly_dates(trading_dates, args.start_date, args.end_date)
        else:
            dates = sample_daily_dates(trading_dates, args.start_date, args.end_date)

        if not dates:
            console.print("[bold red]Error:[/bold red] No trading dates found in the given range.")
            import sys; sys.exit(1)

        if args.resume:
            logs_dir = args.resume.rstrip("/")
            console.print(f"[bold cyan]Resuming run from:[/bold cyan] {logs_dir}")
        else:
            slug = f"{args.start_date}_to_{args.end_date}_{args.freq}_{args.ticker or 'all'}"
            logs_dir = f"logs/{timestamp}_{slug}"
            os.makedirs(logs_dir, exist_ok=True)

        console.print(f"[bold]Running pipeline for {len(dates)} {args.freq} signal dates: {dates[0]} to {dates[-1]} (concurrency={args.concurrency})[/bold]")
        asyncio.run(run_date_range(
            dates,
            ticker=args.ticker,
            base_logs_dir=logs_dir,
            concurrency=args.concurrency,
            run_backtest_flag=args.backtest,
            b_param=args.b_param,
            tcost_bps=args.tcost,
            etf_data_path=args.etf_data,
        ))
