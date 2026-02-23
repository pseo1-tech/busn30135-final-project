import json
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import prompts
from models import ArticleData, ArticleScore, BATCH_SIZE, SentimentState, TickerAggregate
from tools import fetch_benzinga_news, finbert_score_articles
from ui import console

# --- GLOBAL MODEL ---
llm = None


# Provider configurations for OpenAI-compatible API hosts
PROVIDER_CONFIG = {
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "alt_env_key": "GEMINI_API_KEY",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
    },
    "together": {
        "env_key": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-70b-versatile",
    },
    "fireworks": {
        "env_key": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    },
}


def get_model(provider: str, model_name: str, temperature=0):
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider in PROVIDER_CONFIG and "base_url" in PROVIDER_CONFIG[provider]:
        # OpenAI-compatible providers (Together, Groq, Fireworks)
        cfg = PROVIDER_CONFIG[provider]
        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            raise ValueError(f"Missing {cfg['env_key']} environment variable for provider '{provider}'.")
        return ChatOpenAI(
            model=model_name,
            base_url=cfg["base_url"],
            api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: {', '.join(PROVIDER_CONFIG.keys())}")


def verify_model_access(model, provider: str):
    cfg = PROVIDER_CONFIG.get(provider, {})
    env_key = cfg.get("env_key")
    alt_key = cfg.get("alt_env_key")

    if env_key and not os.environ.get(env_key):
        if alt_key and os.environ.get(alt_key):
            pass  # Alt key is set, that's fine
        else:
            raise ValueError(f"Missing {env_key} environment variable.")

    try:
        model.invoke([HumanMessage(content="Hello")])
    except Exception as e:
        raise RuntimeError(f"Model connection failed for '{provider}': {e}")


def configure_agents(model_spec: str = None):
    """
    Configure the global LLM.

    model_spec format: "provider:model_name"
    Examples:
        google:gemini-2.5-flash-lite
        openai:gpt-4o-mini
        together:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
        groq:llama-3.1-70b-versatile
        fireworks:accounts/fireworks/models/llama-v3p1-70b-instruct
    """
    global llm
    llm = None

    using_cli = bool(model_spec)

    try:
        if model_spec:
            provider, model_name = model_spec.split(":", 1)
            llm = get_model(provider, model_name)
            verify_model_access(llm, provider)

        if not llm:
            # Try providers in order: Together > Groq > Google > OpenAI
            # Together/Groq with Llama preferred for backtesting (earlier knowledge cutoff)
            fallback_order = [
                ("together", PROVIDER_CONFIG["together"]["default_model"]),
                ("groq", PROVIDER_CONFIG["groq"]["default_model"]),
                ("google", "gemini-2.5-flash-lite"),
                ("openai", "gpt-4o-mini"),
            ]

            for prov, default_model in fallback_order:
                cfg = PROVIDER_CONFIG.get(prov, {})
                env_key = cfg.get("env_key", "")
                if os.environ.get(env_key):
                    try:
                        llm = get_model(prov, default_model)
                        verify_model_access(llm, prov)
                        console.print(f"[green]Using {prov}:{default_model}[/green]")
                        break
                    except Exception as e:
                        console.print(f"WARN: {prov} failed ({e}), trying next...", style="yellow")
                        llm = None

        if not llm:
            keys = [cfg["env_key"] for cfg in PROVIDER_CONFIG.values() if "env_key" in cfg]
            raise ValueError(f"No valid models found. Set one of: {', '.join(keys)}")

    except Exception as e:
        if using_cli:
            raise e
        console.print(f"WARN: LLM Init failed ({e}).", style="yellow")
        raise


def get_content(response):
    """Helper to safely get content from AIMessage or string."""
    content = response.content if hasattr(response, 'content') else response
    if isinstance(content, list):
        return " ".join([c if isinstance(c, str) else c.get("text", "") for c in content])
    return str(content)


def get_model_name(model) -> str:
    if model is None:
        return "unknown"
    if hasattr(model, 'model_name'):
        return model.model_name
    elif hasattr(model, 'model'):
        return model.model
    return model.__class__.__name__


# --- PRICING (same as lab_01) ---

PRICING = {
    # OpenAI
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-2025-08-07": {"input": 1.25, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Google Gemini
    "gemini-2.5-flash-lite": {"input": 0.01, "output": 0.04},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
    # Llama via Together AI
    "Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "Meta-Llama-3.1-8B-Instruct-Turbo": {"input": 0.18, "output": 0.18},
    # Llama via Groq
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    # Llama via Fireworks
    "llama-v3p1-70b-instruct": {"input": 0.90, "output": 0.90},
    # Fallback
    "unknown": {"input": 0.00, "output": 0.00},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing_key = model_name
    if pricing_key not in PRICING:
        for key in PRICING:
            if key in model_name:
                pricing_key = key
                break
        else:
            pricing_key = "unknown"
    pricing = PRICING[pricing_key]
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def track_llm_call(response, node_name: str, model_name: str, purpose: str = "") -> dict:
    usage = None
    if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
        usage = response.usage_metadata
    elif hasattr(response, 'response_metadata') and response.response_metadata:
        metadata = response.response_metadata
        if 'usage_metadata' in metadata:
            usage = metadata['usage_metadata']
        elif 'token_usage' in metadata:
            usage = metadata['token_usage']

    if usage is None:
        input_tokens = output_tokens = total_tokens = 0
    elif isinstance(usage, dict):
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
    else:
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)

    cost = calculate_cost(model_name, input_tokens, output_tokens)
    return {
        "timestamp": datetime.now().isoformat(),
        "node": node_name,
        "model": model_name,
        "purpose": purpose,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }


def aggregate_node_usage(llm_calls: list, node_name: str) -> dict:
    node_calls = [c for c in llm_calls if c["node"] == node_name]
    if not node_calls:
        return {"num_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}
    return {
        "num_calls": len(node_calls),
        "input_tokens": sum(c["input_tokens"] for c in node_calls),
        "output_tokens": sum(c["output_tokens"] for c in node_calls),
        "total_tokens": sum(c["total_tokens"] for c in node_calls),
        "cost": sum(c["cost"] for c in node_calls),
    }


def log_step(state, filename: str, content: str):
    if hasattr(state, 'logs_dir') and state.logs_dir:
        try:
            with open(f"{state.logs_dir}/{filename}", "w") as f:
                f.write(content)
        except Exception as e:
            console.print(f"WARN: Failed to log {filename}: {e}", style="yellow")


# ============================================================
# PIPELINE NODES
# ============================================================

def run_prompt_optimizer_standalone() -> str:
    """Run the prompt optimizer outside the graph pipeline. Returns the optimized prompt string."""
    console.print("[bold yellow]Generating optimized analyst prompt via meta-prompt...[/bold yellow]")
    messages = [
        SystemMessage(content=prompts.config.meta_prompt),
        HumanMessage(content="Task, Goal, or Current Prompt:\n" + prompts.config.analyst_baseline_for_optimization),
    ]
    response = llm.invoke(messages)
    optimized_prompt = get_content(response).strip()
    console.print("[bold yellow]Optimized analyst prompt generated.[/bold yellow]")
    return optimized_prompt


def prompt_optimizer_node(state: SentimentState):
    """
    Uses the meta-prompt pattern from lab_04 to generate an optimized analyst system prompt.
    The LLM rewrites our baseline analyst prompt into a more effective version.
    Skips generation if an optimized prompt is already cached (e.g. from a prior date in a range).
    """
    if state.optimized_analyst_prompt:
        console.print("[bold yellow]Reusing cached optimized analyst prompt.[/bold yellow]")
        return {}

    console.print("[bold yellow]Generating optimized analyst prompt via meta-prompt...[/bold yellow]")

    messages = [
        SystemMessage(content=prompts.config.meta_prompt),
        HumanMessage(content="Task, Goal, or Current Prompt:\n" + prompts.config.analyst_baseline_for_optimization),
    ]

    response = llm.invoke(messages)
    optimized_prompt = get_content(response).strip()
    usage_record = track_llm_call(response, "prompt_optimizer", get_model_name(llm), "Generate optimized analyst prompt")

    if state.logs_dir:
        log_step(state, "00_optimized_analyst_prompt.txt", optimized_prompt)

    console.print("[bold yellow]Optimized analyst prompt generated.[/bold yellow]")

    return {"optimized_analyst_prompt": optimized_prompt, "llm_calls": [usage_record]}


def news_collector_node(state: SentimentState):
    """Fetches articles from Benzinga and returns them as collected_articles."""
    console.print(f"[bold cyan]Fetching Benzinga news for {state.date_str}...[/bold cyan]")

    articles = fetch_benzinga_news(state.date_str, ticker=state.ticker_filter)

    console.print(f"[bold cyan]Found {len(articles)} articles.[/bold cyan]")

    if state.logs_dir:
        log_step(state, "00_collected_articles.json", json.dumps(
            [a.model_dump() for a in articles], indent=2, default=str
        ))

    return {"collected_articles": articles}


def finbert_scorer_node(state: SentimentState):
    """
    Runs FinBERT on all collected articles to pre-compute sentiment scores.
    This is fast (~50ms per article on CPU) and reduces the LLM's workload.
    """
    articles = state.collected_articles
    if not articles:
        return {}

    console.print(f"[bold green]Running FinBERT on {len(articles)} articles...[/bold green]")

    scores = finbert_score_articles(articles)

    # Update articles with FinBERT scores
    updated_articles = []
    for article, fb_score in zip(articles, scores):
        updated = article.model_copy(update={"finbert_sentiment": fb_score})
        updated_articles.append(updated)

    console.print(f"[bold green]FinBERT scoring complete. Avg sentiment: {sum(scores)/len(scores):+.3f}[/bold green]")

    if state.logs_dir:
        log_step(state, "01_finbert_scores.json", json.dumps(
            [{"title": a.title[:60], "finbert_sentiment": s} for a, s in zip(articles, scores)], indent=2
        ))

    return {"collected_articles": updated_articles}


def dispatcher_node(state: SentimentState):
    """
    Logs dispatch info. Fan-out into batches is handled by the graph's conditional edge.
    """
    num = len(state.collected_articles)
    num_batches = (num + BATCH_SIZE - 1) // BATCH_SIZE
    tickers_seen = set()
    for a in state.collected_articles:
        tickers_seen.update(a.tickers)

    console.print(
        f"[bold magenta]Dispatching {num} articles in {num_batches} batches "
        f"(batch size {BATCH_SIZE}). Tickers: {', '.join(sorted(tickers_seen)) or 'none'}[/bold magenta]"
    )

    return {}  # State unchanged; fan-out is handled by conditional edge in graph.py


def analyst_node(state_subset: dict):
    """
    Scores a BATCH of articles in a single LLM call. Called in parallel via Send.
    Receives a dict with 'articles' (list of ArticleData dicts), 'logs_dir',
    'optimized_analyst_prompt', and 'batch_index'.

    Each article already has a finbert_sentiment pre-score from the FinBERT node.
    """
    article_dicts = state_subset["articles"]
    logs_dir = state_subset.get("logs_dir")
    optimized_prompt = state_subset.get("optimized_analyst_prompt")
    batch_idx = state_subset.get("batch_index", 0)

    articles = [ArticleData(**a) if isinstance(a, dict) else a for a in article_dicts]

    # Use optimized prompt as system message if available, otherwise fall back to default
    system_msg = optimized_prompt if optimized_prompt else prompts.config.analyst_system

    # Build the batched articles block
    articles_block = ""
    for i, article in enumerate(articles):
        articles_block += f"""--- Article {i} ---
TITLE: {article.title}
TICKERS: {", ".join(article.tickers) if article.tickers else "None mentioned"}
FINBERT SENTIMENT: {article.finbert_sentiment if article.finbert_sentiment is not None else "N/A"}
BODY (truncated):
{article.body[:3000]}

"""

    user_prompt = prompts.config.analyst_batch_main.format(articles_block=articles_block)

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content = get_content(response)
    usage_record = track_llm_call(response, "analyst", get_model_name(llm), f"Batch {batch_idx}: {len(articles)} articles")

    # Parse JSON array response
    scores_out = []
    try:
        content_clean = content.replace("```json", "").replace("```", "").strip()
        data_list = json.loads(content_clean)
        if not isinstance(data_list, list):
            data_list = [data_list]

        for i, article in enumerate(articles):
            # Match by article_index or fall back to position
            data = next((d for d in data_list if d.get("article_index") == i), None)
            if data is None and i < len(data_list):
                data = data_list[i]

            if data:
                score = ArticleScore(
                    benzinga_id=article.benzinga_id,
                    title=article.title,
                    tickers=article.tickers,
                    sentiment_score=max(-1.0, min(1.0, float(data.get("sentiment_score", 0)))),
                    price_direction_score=max(1, min(10, int(data.get("price_direction_score", 5)))),
                    reasoning=data.get("reasoning", ""),
                )
            else:
                # Fallback: use FinBERT score directly
                score = ArticleScore(
                    benzinga_id=article.benzinga_id,
                    title=article.title,
                    tickers=article.tickers,
                    sentiment_score=article.finbert_sentiment or 0.0,
                    price_direction_score=5,
                    reasoning="LLM did not return a score for this article; using FinBERT fallback.",
                )
            scores_out.append(score)

    except Exception as e:
        console.print(f"WARN: Failed to parse batch {batch_idx} response: {e}", style="yellow")
        # Fallback: use FinBERT scores for all articles in this batch
        for article in articles:
            scores_out.append(ArticleScore(
                benzinga_id=article.benzinga_id,
                title=article.title,
                tickers=article.tickers,
                sentiment_score=article.finbert_sentiment or 0.0,
                price_direction_score=5,
                reasoning=f"Batch parse error: {e}. Using FinBERT fallback.",
            ))

    if logs_dir:
        try:
            with open(f"{logs_dir}/02_analyst_batch_{batch_idx}.json", "w") as f:
                json.dump([s.model_dump() for s in scores_out], f, indent=2)
        except Exception:
            pass

    return {
        "article_scores": [s.model_dump() for s in scores_out],
        "llm_calls": [usage_record],
    }


def aggregator_node(state: SentimentState):
    """Groups article scores by ticker, computes averages, and generates overall reasoning."""
    # Group by ticker
    ticker_map: dict[str, list[ArticleScore]] = {}
    for score_dict in state.article_scores:
        score = ArticleScore(**score_dict)
        for ticker in score.tickers:
            ticker_map.setdefault(ticker, []).append(score)

    # Also group articles with no tickers under "GENERAL"
    for score_dict in state.article_scores:
        score = ArticleScore(**score_dict)
        if not score.tickers:
            ticker_map.setdefault("GENERAL", []).append(score)

    aggregates = []

    for ticker, scores in sorted(ticker_map.items()):
        avg_sent = sum(s.sentiment_score for s in scores) / len(scores)
        avg_dir = sum(s.price_direction_score for s in scores) / len(scores)

        aggregates.append(TickerAggregate(
            ticker=ticker,
            num_articles=len(scores),
            avg_sentiment=round(avg_sent, 3),
            avg_price_direction=round(avg_dir, 1),
            article_scores=scores,
        ))

    console.print(f"[bold cyan]Aggregated {len(state.article_scores)} scores across {len(aggregates)} tickers.[/bold cyan]")

    if state.logs_dir:
        log_step(state, "03_aggregates.json", json.dumps(
            [a.model_dump() for a in aggregates], indent=2
        ))

    return {"ticker_aggregates": aggregates}
