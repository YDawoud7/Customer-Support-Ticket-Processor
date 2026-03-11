"""Tests for the FastAPI application endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph import build_graph
from src.state import QualityAssessment, TicketClassification


def _build_mocked_graph(category="billing", confidence=0.9, quality_approved=True, checkpointer=None):
    """Build a mocked graph for API testing."""
    mock_classifier_structured = MagicMock()
    mock_classifier_structured.invoke.return_value = TicketClassification(
        category=category,
        confidence=confidence,
        reasoning=f"Mock classified as {category}",
    )

    mock_quality_structured = MagicMock()
    mock_quality_structured.invoke.return_value = QualityAssessment(
        approved=quality_approved,
        feedback="Mock approved." if quality_approved else "Mock rejected: needs work.",
    )

    call_count = {"n": 0}

    def chat_model_factory(*args, **kwargs):
        mock = MagicMock()
        call_count["n"] += 1
        n = call_count["n"]
        if n == 1:
            mock.with_structured_output.return_value = mock_classifier_structured
        elif n <= 4:
            agent_model = MagicMock()
            agent_model.invoke.return_value = AIMessage(
                content=f"Mock {category} agent response.", tool_calls=[]
            )
            mock.bind_tools.return_value = agent_model
            mock.invoke.return_value = AIMessage(
                content=f"Mock {category} agent response.", tool_calls=[]
            )
        else:
            mock.with_structured_output.return_value = mock_quality_structured
        return mock

    mock_tool = MagicMock()
    mock_tool.name = "mock_tool"

    patches = [
        patch("src.classifier.create_chat_model", side_effect=chat_model_factory),
        patch("src.agents.create_chat_model", side_effect=chat_model_factory),
        patch("src.quality_check.create_chat_model", side_effect=chat_model_factory),
        patch("src.graph.create_search_billing_docs", return_value=mock_tool),
        patch("src.graph.create_search_technical_docs", return_value=mock_tool),
        patch("src.graph.create_search_general_docs", return_value=mock_tool),
    ]
    for p in patches:
        p.start()
    try:
        graph = build_graph(checkpointer=checkpointer)
    finally:
        for p in reversed(patches):
            p.stop()
    return graph


def _create_test_client(**kwargs):
    """Create a TestClient with a mocked graph."""
    import api

    checkpointer = MemorySaver()
    graph = _build_mocked_graph(checkpointer=checkpointer, **kwargs)
    api.graph_app = graph
    api.checkpointer = checkpointer
    return TestClient(api.app)


class TestHealthEndpoint:
    def test_returns_ok(self):
        client = _create_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestSubmitTicket:
    def test_completes_successfully(self):
        client = _create_test_client()
        resp = client.post("/tickets", json={"ticket_text": "I was charged twice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["category"] == "billing"
        assert data["confidence"] == 0.9
        assert len(data["response"]) > 0
        assert data["quality_approved"] is True
        assert data["thread_id"]

    def test_returns_interrupt_on_low_confidence(self):
        client = _create_test_client(confidence=0.3)
        resp = client.post("/tickets", json={"ticket_text": "Unclear ticket"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "interrupted"
        assert data["interrupt"] is not None
        assert data["interrupt"]["type"] == "confidence"

    def test_returns_interrupt_on_quality_failure(self):
        client = _create_test_client(quality_approved=False)
        resp = client.post("/tickets", json={"ticket_text": "Billing issue"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "interrupted"
        assert data["interrupt"] is not None
        assert data["interrupt"]["type"] == "quality"

    def test_invalid_model_config(self):
        client = _create_test_client()
        resp = client.post(
            "/tickets",
            json={"ticket_text": "Test", "model_config_name": "nonexistent"},
        )
        assert resp.status_code == 400


class TestResumeTicket:
    def test_resume_after_confidence_interrupt(self):
        client = _create_test_client(confidence=0.3)
        resp = client.post("/tickets", json={"ticket_text": "Unclear ticket"})
        thread_id = resp.json()["thread_id"]

        resp = client.post(
            f"/tickets/{thread_id}/resume",
            json={"value": "billing"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "billing"

    def test_resume_nonexistent_thread(self):
        client = _create_test_client()
        resp = client.post(
            "/tickets/nonexistent-id/resume",
            json={"value": "approve"},
        )
        assert resp.status_code == 404


class TestGetTicket:
    def test_get_completed_ticket(self):
        client = _create_test_client()
        resp = client.post("/tickets", json={"ticket_text": "Billing question"})
        thread_id = resp.json()["thread_id"]

        resp = client.get(f"/tickets/{thread_id}")
        assert resp.status_code == 200
        assert resp.json()["category"] == "billing"

    def test_get_nonexistent_ticket(self):
        client = _create_test_client()
        resp = client.get("/tickets/nonexistent-id")
        assert resp.status_code == 404


class TestStreamEndpoint:
    def test_stream_returns_sse_events(self):
        client = _create_test_client()
        with client.stream("POST", "/tickets/stream", json={"ticket_text": "Billing issue"}) as resp:
            assert resp.status_code == 200
            events = []
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
            assert "start" in events
            assert "complete" in events
