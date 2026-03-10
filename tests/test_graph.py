from unittest.mock import MagicMock, patch

from src.graph import build_graph, route_by_category
from src.state import TicketClassification


def _mock_classifier_returning(category: str, confidence: float = 0.9):
    """Create a mock classifier node that sets the given category."""

    def fake_classifier(state):
        return {
            "category": category,
            "confidence": confidence,
            "reasoning": f"Mock classified as {category}",
            "messages": [
                {"role": "assistant", "content": f"Classified as {category}"}
            ],
        }

    return fake_classifier


def _build_graph_with_mock_classifier(category: str):
    """Build the full graph but replace the classifier with a mock."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = TicketClassification(
        category=category,
        confidence=0.9,
        reasoning=f"Mock classified as {category}",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    with patch("src.classifier.ChatAnthropic", return_value=mock_model):
        return build_graph()


def _invoke_with_ticket(graph, ticket_text: str) -> dict:
    return graph.invoke(
        {
            "messages": [],
            "ticket_text": ticket_text,
            "category": "",
            "confidence": 0.0,
            "reasoning": "",
            "response": "",
        }
    )


class TestRouteByCategory:
    def test_routes_billing(self):
        assert route_by_category({"category": "billing"}) == "billing"

    def test_routes_technical(self):
        assert route_by_category({"category": "technical"}) == "technical"

    def test_routes_general(self):
        assert route_by_category({"category": "general"}) == "general"


class TestGraphEndToEnd:
    def test_billing_ticket_routes_to_billing_agent(self):
        graph = _build_graph_with_mock_classifier("billing")
        result = _invoke_with_ticket(graph, "I was charged twice.")
        assert result["category"] == "billing"
        assert "Billing Agent" in result["response"]

    def test_technical_ticket_routes_to_technical_agent(self):
        graph = _build_graph_with_mock_classifier("technical")
        result = _invoke_with_ticket(graph, "App crashes on startup.")
        assert result["category"] == "technical"
        assert "Technical Agent" in result["response"]

    def test_general_ticket_routes_to_general_agent(self):
        graph = _build_graph_with_mock_classifier("general")
        result = _invoke_with_ticket(graph, "What are your hours?")
        assert result["category"] == "general"
        assert "General Agent" in result["response"]

    def test_state_has_all_fields_after_run(self):
        graph = _build_graph_with_mock_classifier("billing")
        result = _invoke_with_ticket(graph, "Refund my order.")
        assert result["ticket_text"] == "Refund my order."
        assert result["confidence"] == 0.9
        assert len(result["reasoning"]) > 0
        assert len(result["response"]) > 0
        assert len(result["messages"]) > 0
