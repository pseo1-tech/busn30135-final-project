from pydantic import BaseModel


class PromptConfig(BaseModel):
    # --- Analyst (per-article scoring) ---
    analyst_system: str = "You are a financial analyst. You output JSON only."

    analyst_main: str = """You are a Senior Financial Analyst specializing in news-driven sentiment analysis.

Analyze the following news article and produce TWO scores:

1. **sentiment_score** (float, -1.0 to +1.0):
   - -1.0 = extremely negative sentiment
   - 0.0 = neutral
   - +1.0 = extremely positive sentiment

2. **price_direction_score** (integer, 1 to 10):
   - 1 = strong sell signal (stock price very likely to decrease)
   - 5 = neutral (no clear direction)
   - 10 = strong buy signal (stock price very likely to increase)

3. **reasoning**: A 2-3 sentence explanation of your scores.

Consider: tone of language, implications for company revenue/growth, regulatory risk, market reaction signals, and broader sector impact.

ARTICLE TITLE: {title}

TICKERS MENTIONED: {tickers}

ARTICLE BODY:
{body}

Output ONLY valid JSON in this exact format:
{{
    "sentiment_score": <float between -1.0 and 1.0>,
    "price_direction_score": <integer between 1 and 10>,
    "reasoning": "<brief explanation>"
}}
"""

    # --- Batch Analyst (multiple articles per call, with FinBERT pre-scores) ---
    analyst_batch_main: str = """You are a Senior Financial Analyst. You have been given a batch of news articles, each with a pre-computed FinBERT sentiment score.

For EACH article, produce:
1. **sentiment_score** (float, -1.0 to +1.0): Use the FinBERT score as a starting point but adjust based on your deeper reading of the article content, context, and financial implications.
2. **price_direction_score** (integer, 1 to 10): 1 = strong sell signal, 5 = neutral, 10 = strong buy signal. This is your independent assessment of near-term price direction.
3. **reasoning**: A 1-2 sentence explanation.

ARTICLES:
{articles_block}

Output ONLY a valid JSON array with one object per article, in the same order:
[
    {{
        "article_index": 0,
        "sentiment_score": <float between -1.0 and 1.0>,
        "price_direction_score": <integer between 1 and 10>,
        "reasoning": "<brief explanation>"
    }},
    ...
]
"""

    # --- Meta-Prompt (used to generate optimized analyst prompt) ---
    # Adapted from lab_04 Section 3.2 (OpenAI prompt generation pattern)
    meta_prompt: str = """Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions: Encourage reasoning steps before any conclusions are reached. NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection.
- Output Format: Explicitly state the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt.

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are.]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
"""

    analyst_baseline_for_optimization: str = """Your job is to analyze a BATCH of news articles about publicly traded companies and produce scores for each.

Each article comes with a pre-computed FinBERT sentiment score (-1.0 to +1.0) from a financial sentiment BERT model. Use this as a starting point but adjust based on your deeper understanding.

For each article, produce:
1. sentiment_score: a float from -1.0 (extremely negative) to +1.0 (extremely positive) reflecting the overall sentiment. Use the FinBERT score as an anchor but refine it.
2. price_direction_score: an integer from 1 (strong sell signal, price very likely to decrease) to 10 (strong buy signal, price very likely to increase) predicting the near-term stock price direction.
3. reasoning: a 1-2 sentence explanation justifying the scores.

Consider: tone of language, implications for company revenue/growth, regulatory risk, market reaction signals, competitive positioning, and broader sector impact.

Each article includes: index, title, tickers, FinBERT score, and truncated body text.

Output must be a valid JSON array with one object per article:
[{{"article_index": 0, "sentiment_score": <float>, "price_direction_score": <integer>, "reasoning": "<string>"}}, ...]
"""

    # --- Aggregator (per-ticker summary) ---
    aggregator_system: str = "You are a financial analyst. You output JSON only."

    aggregator_main: str = """You are a Chief Market Strategist synthesizing multiple news signals for a single stock ticker.

TICKER: {ticker}

Below are the individual article scores for this ticker today:

{article_summaries}

Based on these signals, provide an overall assessment.

Output ONLY valid JSON:
{{
    "overall_reasoning": "<2-3 sentence synthesis of what the news means for this ticker's near-term price action>"
}}
"""


# Global instance
config = PromptConfig()


def update_prompts(new_config: dict):
    """Updates the global config with values from a dict."""
    global config
    current_dump = config.model_dump()
    current_dump.update(new_config)
    config = PromptConfig(**current_dump)


def load_from_file(filepath: str):
    """Loads prompt config from a JSON file."""
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
        update_prompts(data)
