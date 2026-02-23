import os
from typing import List, Optional

from bs4 import BeautifulSoup
from massive import RESTClient

from models import ArticleData

# --- FinBERT singleton (lazy-loaded) ---
_finbert_pipeline = None


def get_finbert():
    """Lazy-load the FinBERT pipeline. Downloads model on first use (~420MB)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU; change to 0 for GPU
        )
    return _finbert_pipeline


def finbert_score_articles(articles: List[ArticleData]) -> List[float]:
    """
    Score a list of articles using FinBERT. Returns sentiment scores in [-1, +1].

    FinBERT outputs one of: positive, negative, neutral with a confidence.
    We map: positive -> +confidence, negative -> -confidence, neutral -> 0.
    Uses the article title + first 512 chars of body (FinBERT's max input is 512 tokens).
    """
    if not articles:
        return []

    finbert = get_finbert()

    # Prepare inputs: title + truncated body
    texts = [f"{a.title}. {a.body[:512]}" for a in articles]

    # FinBERT batch inference
    results = finbert(texts, batch_size=32, truncation=True, max_length=512)

    scores = []
    for r in results:
        label = r["label"].lower()
        confidence = r["score"]
        if label == "positive":
            scores.append(round(confidence, 4))
        elif label == "negative":
            scores.append(round(-confidence, 4))
        else:  # neutral
            scores.append(0.0)

    return scores


def strip_html(html_content: str) -> str:
    """Strip HTML tags and clean up whitespace from article body."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


def fetch_benzinga_news(date_str: str, ticker: Optional[str] = None, api_key: Optional[str] = None) -> List[ArticleData]:
    """
    Fetch news articles from Benzinga via the massive package.

    Args:
        date_str: Date to query (e.g. '2025-06-01')
        ticker: Optional ticker symbol to filter articles
        api_key: Massive API key. Falls back to MASSIVE_API_KEY env var.

    Returns:
        List of ArticleData with HTML-stripped body text.
    """
    key = api_key or os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise ValueError("No API key provided. Set MASSIVE_API_KEY env var or pass api_key argument.")

    client = RESTClient(api_key=key)

    articles = []
    for item in client.list_benzinga_news_v2(published=date_str):
        item_tickers = item.tickers or []

        # If ticker filter is set, skip articles that don't mention it
        if ticker and ticker.upper() not in [t.upper() for t in item_tickers]:
            continue

        body_text = strip_html(item.body or "")
        if not body_text:
            continue  # Skip empty articles

        articles.append(ArticleData(
            title=item.title or "Untitled",
            body=body_text[:10000],  # Cap at 10k chars to manage context
            tickers=item_tickers,
            published=item.published,
            url=item.url,
            benzinga_id=item.benzinga_id,
        ))

    return articles
