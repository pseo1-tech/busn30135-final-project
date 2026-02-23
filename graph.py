from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from agents import aggregator_node, analyst_node, dispatcher_node, finbert_scorer_node, news_collector_node, prompt_optimizer_node
from models import BATCH_SIZE, SentimentState


def route_analysis(state: SentimentState):
    """Fan-out: Send batches of articles to analyst nodes in parallel, including the optimized prompt."""
    articles = state.collected_articles
    batches = [articles[i:i + BATCH_SIZE] for i in range(0, len(articles), BATCH_SIZE)]

    return [
        Send("analyst", {
            "articles": [a.model_dump() for a in batch],
            "logs_dir": state.logs_dir,
            "optimized_analyst_prompt": state.optimized_analyst_prompt,
            "batch_index": idx,
        })
        for idx, batch in enumerate(batches)
    ]


# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(SentimentState)

# Add Nodes
workflow.add_node("news_collector", news_collector_node)
workflow.add_node("finbert_scorer", finbert_scorer_node)
workflow.add_node("prompt_optimizer", prompt_optimizer_node)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("aggregator", aggregator_node)

# Add Edges
# Flow: news_collector → finbert_scorer → prompt_optimizer → dispatcher → analyst (batched, parallel) → aggregator → END
workflow.set_entry_point("news_collector")
workflow.add_edge("news_collector", "finbert_scorer")
workflow.add_edge("finbert_scorer", "prompt_optimizer")
workflow.add_edge("prompt_optimizer", "dispatcher")

# Dispatcher fans out to analysts (one per batch)
workflow.add_conditional_edges("dispatcher", route_analysis, ["analyst"])

# All analyst batches feed into aggregator
workflow.add_edge("analyst", "aggregator")

# Aggregator is the final node
workflow.add_edge("aggregator", END)

# Compile
app = workflow.compile()
