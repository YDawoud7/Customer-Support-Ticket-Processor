from langgraph.graph import END, START, StateGraph

from src.agents import billing_agent, general_agent, technical_agent
from src.classifier import create_classifier
from src.state import TicketState


def route_by_category(state: TicketState) -> str:
    """Conditional edge: route to the specialist agent matching the category."""
    return state["category"]


def build_graph(classifier_model: str = "claude-sonnet-4-20250514"):
    """Construct and compile the ticket processing graph.

    Graph flow:
        START → classifier → (billing | technical | general) → END
    """
    graph = StateGraph(TicketState)

    graph.add_node("classifier", create_classifier(model_name=classifier_model))
    graph.add_node("billing", billing_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("general", general_agent)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_by_category,
        {"billing": "billing", "technical": "technical", "general": "general"},
    )
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("general", END)

    return graph.compile()
