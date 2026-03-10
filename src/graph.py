from langgraph.graph import END, START, StateGraph

from src.agents import create_billing_agent, create_general_agent, create_technical_agent
from src.classifier import create_classifier
from src.confidence_gate import confidence_gate
from src.models import ALL_CLAUDE, ModelConfig
from src.quality_check import create_quality_check
from src.quality_gate import quality_gate
from src.state import TicketState
from src.tools.calculator import calculator
from src.tools.code_analysis import analyze_code
from src.tools.search import (
    create_search_billing_docs,
    create_search_general_docs,
    create_search_technical_docs,
)


def route_by_category(state: TicketState) -> str:
    """Conditional edge: route to the specialist agent matching the category."""
    return state["category"]


def build_graph(model_config: ModelConfig | None = None, checkpointer=None):
    """Construct and compile the ticket processing graph.

    Graph flow:
        START → classifier → confidence_gate → (billing | technical | general)
              → quality_check → quality_gate → END
    """
    config = {**ALL_CLAUDE, **(model_config or {})}

    billing_tools = [create_search_billing_docs(), calculator]
    technical_tools = [create_search_technical_docs(), analyze_code]
    general_tools = [create_search_general_docs()]

    graph = StateGraph(TicketState)

    graph.add_node("classifier", create_classifier(model_name=config["classifier"]))
    graph.add_node("confidence_gate", confidence_gate)
    graph.add_node(
        "billing",
        create_billing_agent(model_name=config["billing"], tools=billing_tools),
    )
    graph.add_node(
        "technical",
        create_technical_agent(model_name=config["technical"], tools=technical_tools),
    )
    graph.add_node(
        "general",
        create_general_agent(model_name=config["general"], tools=general_tools),
    )
    graph.add_node("quality_check", create_quality_check(model_name=config["quality_check"]))
    graph.add_node("quality_gate", quality_gate)

    graph.add_edge(START, "classifier")
    graph.add_edge("classifier", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        route_by_category,
        {"billing": "billing", "technical": "technical", "general": "general"},
    )
    graph.add_edge("billing", "quality_check")
    graph.add_edge("technical", "quality_check")
    graph.add_edge("general", "quality_check")
    graph.add_edge("quality_check", "quality_gate")
    graph.add_edge("quality_gate", END)

    return graph.compile(checkpointer=checkpointer)
