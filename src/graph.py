from langgraph.graph import END, START, StateGraph

from src.agents import create_billing_agent, create_general_agent, create_technical_agent
from src.classifier import create_classifier
from src.confidence_gate import confidence_gate
from src.quality_check import create_quality_check
from src.state import TicketState
from src.vector_store import get_retriever


def route_by_category(state: TicketState) -> str:
    """Conditional edge: route to the specialist agent matching the category."""
    return state["category"]


def build_graph(classifier_model: str = "claude-sonnet-4-20250514", checkpointer=None):
    """Construct and compile the ticket processing graph.

    Graph flow:
        START → classifier → confidence_gate → (billing | technical | general)
              → quality_check → END
    """
    billing_retriever = get_retriever("billing_docs")
    technical_retriever = get_retriever("technical_docs")

    graph = StateGraph(TicketState)

    graph.add_node("classifier", create_classifier(model_name=classifier_model))
    graph.add_node("confidence_gate", confidence_gate)
    graph.add_node(
        "billing",
        create_billing_agent(model_name=classifier_model, retriever=billing_retriever),
    )
    graph.add_node(
        "technical",
        create_technical_agent(
            model_name=classifier_model, retriever=technical_retriever
        ),
    )
    graph.add_node("general", create_general_agent(model_name=classifier_model))
    graph.add_node("quality_check", create_quality_check(model_name=classifier_model))

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
    graph.add_edge("quality_check", END)

    return graph.compile(checkpointer=checkpointer)
