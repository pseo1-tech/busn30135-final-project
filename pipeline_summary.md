# News Sentiment & Price Direction Scoring Agent -- Pipeline Summary

## What We Built
A multi-agent AI pipeline that automatically scores news articles for **sentiment** and **predicted stock price direction**, then runs a **sector rotation backtest** using those signals. Built on the LangGraph framework from our course labs.

## Pipeline Architecture

```
NewsCollector -> FinBERT Scorer -> PromptOptimizer -> Dispatcher -> Analyst (batched, parallel) -> Aggregator -> END
```

**6 agent nodes**, each with a specific role:

| Node | Role |
|------|------|
| **NewsCollector** | Pulls articles from Benzinga via the `massive` API, strips HTML, extracts tickers |
| **FinBERT Scorer** | Runs ProsusAI/finbert locally on all articles for fast pre-scoring (~50ms/article) |
| **PromptOptimizer** | Uses a meta-prompt (from lab_04) to have the LLM generate its own optimized scoring instructions |
| **Dispatcher** | Fans out articles into batches of 10 for parallel LLM scoring |
| **Analyst** | Scores a batch of articles: **sentiment** (-1.0 to +1.0) and **price direction** (1-10 scale), using FinBERT pre-scores as anchors |
| **Aggregator** | Groups scores by ticker, computes article-weighted averages (pure arithmetic, no LLM) |

## Key Design Decisions

### 1. Hybrid FinBERT + LLM Scoring
FinBERT provides fast local sentiment pre-scores. The LLM refines these with deeper financial reasoning. If the LLM fails to parse, FinBERT scores are used as fallback.

### 2. Batched Parallel Fan-Out
Articles are grouped into batches of 10 and scored in a single LLM call per batch (vs. one call per article). Combined with parallel execution via LangGraph's `Send` API.

### 3. Meta-Prompt Optimization
The LLM rewrites its own scoring prompt at startup (pattern from lab_04 Section 3.2). Generated once and cached across all dates in range mode.

### 4. Backtesting-Safe Model Selection
Supports **Llama 3.1-70B** via Together AI / Groq with a **December 2023 knowledge cutoff** -- no knowledge of 2024 events, giving a clean backtesting window. Gemini/GPT alternatives also supported.

### 5. Weekly Signal Frequency
Signal dates are sampled from actual trading days in the ETF parquet (last trading day per week, typically Friday). This reduces pipeline runs from ~252/year to ~52/year while maintaining meaningful signal frequency.

## Sector Rotation Backtest

### Signal Construction
1. Pipeline scores individual stock tickers from Benzinga news
2. Tickers are mapped to GICS sectors via a static S&P 500 CSV + yfinance fallback
3. Per-ticker scores are aggregated to sector-level signals (article-weighted averages)
4. Sector signals drive portfolio weights via softmax: w_i = exp(b * signal_i) / sum_j exp(b * signal_j)

### Sector ETFs (11 GICS sectors)
| ETF | Sector |
|-----|--------|
| XLK | Technology |
| XLF | Financials |
| XLE | Energy |
| XLV | Health Care |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLI | Industrials |
| XLB | Materials |
| XLU | Utilities |
| XLRE | Real Estate |
| XLC | Communication Services |

### Backtest Methodology
- **Signal date T** (e.g., Friday Jan 10): Run sentiment pipeline, compute sector weights
- **Holding period T to T+1**: Apply weights to actual ETF returns from CRSP data
- **Benchmark**: Equal-weight across all 11 sector ETFs
- **Metrics**: Total return, annualized return/vol, Sharpe ratio, max drawdown

## Speed Optimizations
| Optimization | Impact |
|-------------|--------|
| Weekly frequency (default) | ~5x fewer pipeline runs vs. daily |
| FinBERT pre-scoring | Fast local inference, reduces LLM workload |
| Batched LLM calls (10 articles/call) | ~10x fewer API calls |
| Parallel date processing (asyncio) | Multiple dates scored concurrently |
| Cached optimized prompt | Generated once, reused across all dates |
| Arithmetic-only aggregator | No LLM calls for per-ticker aggregation |

## Output Files
| File | Contents |
|------|----------|
| `scores.csv` | Per (date, ticker) sentiment and direction scores |
| `sector_signals.csv` | Per (date, sector ETF) aggregated signals |
| `backtest_results.csv` | Period-by-period strategy vs. benchmark returns + weights |

## How to Run

```bash
# Set API keys
export TOGETHER_API_KEY="your-key"
export MASSIVE_API_KEY="your-key"

# Weekly backtest over 3 months (default frequency)
python main.py --start-date 2024-06-01 --end-date 2024-08-31 --backtest

# Use sentiment as signal with aggressive tilting
python main.py --start-date 2024-06-01 --end-date 2024-08-31 \
  --backtest --signal avg_sentiment --b-param 10

# Single date scoring
python main.py --date 2024-06-15

# Daily frequency (expensive)
python main.py --start-date 2024-06-01 --end-date 2024-08-31 --freq daily --backtest
```

### CLI Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--freq` | weekly | Signal frequency: weekly or daily |
| `--backtest` | off | Run sector rotation backtest after scoring |
| `--signal` | avg_direction | Signal column: avg_direction or avg_sentiment |
| `--b-param` | 5.0 | Softmax aggressiveness (higher = more concentrated) |
| `--concurrency` | 5 | Max parallel dates |
| `--model` | auto | LLM provider:model spec |

## Files (all in `final_project/`)

| File | Purpose |
|------|---------|
| `models.py` | Pydantic data models (ArticleData, ArticleScore, TickerAggregate, SentimentState) |
| `agents.py` | All 6 agent nodes + multi-provider LLM config + usage/cost tracking |
| `graph.py` | LangGraph StateGraph wiring with batched parallel fan-out |
| `prompts.py` | All prompts including meta-prompt for optimization |
| `tools.py` | Benzinga API wrapper + FinBERT scoring |
| `sector_mapping.py` | Ticker-to-GICS-sector mapping with yfinance fallback |
| `sector_map.csv` | Seed CSV of S&P 500 tickers to sectors (auto-expanding) |
| `backtest.py` | Softmax sector rotation strategy + backtest engine |
| `main.py` | CLI entry point with Rich UI, date range support, backtest integration |
| `ui.py` | Shared Rich console |
