from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph import build_graph, route_by_category
from src.state import QualityAssessment, TicketClassification


def _build_mocked_graph(category: str, confidence: float = 0.9, checkpointer=None, quality_approved=True):
    """Build the full graph with all LLMs and tool creation mocked."""
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
            # Classifier
            mock.with_structured_output.return_value = mock_classifier_structured
        elif n <= 4:
            # Agents — bind_tools returns a model whose invoke returns a text response
            agent_model = MagicMock()
            agent_model.invoke.return_value = AIMessage(
                content=f"Mock {category} agent response.", tool_calls=[]
            )
            mock.bind_tools.return_value = agent_model
            mock.invoke.return_value = AIMessage(
                content=f"Mock {category} agent response.", tool_calls=[]
            )
        else:
            # Quality check
            mock.with_structured_output.return_value = mock_quality_structured
        return mock

    # Mock the tool creation functions to return simple mock tools
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


INITIAL_STATE = {
    "messages": [],
    "ticket_text": "Test ticket",
    "category": "",
    "confidence": 0.0,
    "reasoning": "",
    "response": "",
    "retrieved_docs": [],
    "quality_approved": False,
    "quality_feedback": "",
}


class TestRouteByCategory:
    def test_routes_billing(self):
        assert route_by_category({"category": "billing"}) == "billing"

    def test_routes_technical(self):
        assert route_by_category({"category": "technical"}) == "technical"

    def test_routes_general(self):
        assert route_by_category({"category": "general"}) == "general"


class TestGraphEndToEnd:
    def test_billing_ticket_completes(self):
        graph = _build_mocked_graph("billing")
        result = graph.invoke({**INITIAL_STATE, "ticket_text": "I was charged twice."})
        assert result["category"] == "billing"
        assert len(result["response"]) > 0
        assert result["quality_approved"] is True

    def test_technical_ticket_completes(self):
        graph = _build_mocked_graph("technical")
        result = graph.invoke(
            {**INITIAL_STATE, "ticket_text": "App crashes on startup."}
        )
        assert result["category"] == "technical"
        assert len(result["response"]) > 0

    def test_general_ticket_completes(self):
        graph = _build_mocked_graph("general")
        result = graph.invoke(
            {**INITIAL_STATE, "ticket_text": "What are your hours?"}
        )
        assert result["category"] == "general"
        assert len(result["response"]) > 0

    def test_state_has_all_fields_after_run(self):
        graph = _build_mocked_graph("billing")
        result = graph.invoke({**INITIAL_STATE, "ticket_text": "Refund my order."})
        assert result["ticket_text"] == "Refund my order."
        assert result["confidence"] == 0.9
        assert len(result["reasoning"]) > 0
        assert len(result["response"]) > 0
        assert isinstance(result["quality_approved"], bool)
        assert len(result["quality_feedback"]) > 0
        assert len(result["messages"]) > 0


class TestConfidenceGateIntegration:
    def test_high_confidence_flows_through(self):
        graph = _build_mocked_graph("billing", confidence=0.9)
        result = graph.invoke({**INITIAL_STATE, "ticket_text": "Charge issue"})
        assert result["category"] == "billing"
        assert len(result["response"]) > 0

    def test_low_confidence_pauses_at_interrupt(self):
        checkpointer = MemorySaver()
        graph = _build_mocked_graph("general", confidence=0.3, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-interrupt"}}

        graph.invoke({**INITIAL_STATE, "ticket_text": "Unclear ticket"}, config=config)

        state = graph.get_state(config)
        assert any(
            hasattr(t, "interrupts") and t.interrupts for t in state.tasks
        )

    def test_resume_after_interrupt_completes_flow(self):
        checkpointer = MemorySaver()
        graph = _build_mocked_graph("general", confidence=0.3, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-resume"}}

        graph.invoke({**INITIAL_STATE, "ticket_text": "Unclear ticket"}, config=config)

        result = graph.invoke(Command(resume="general"), config=config)
        assert result["category"] == "general"
        assert len(result["response"]) > 0
        assert result["quality_approved"] is True

    def test_override_category_on_resume(self):
        checkpointer = MemorySaver()
        graph = _build_mocked_graph("general", confidence=0.3, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-override"}}

        graph.invoke({**INITIAL_STATE, "ticket_text": "Unclear ticket"}, config=config)

        result = graph.invoke(Command(resume="billing"), config=config)
        assert result["category"] == "billing"


class TestQualityGateIntegration:
    def test_approved_flows_through(self):
        graph = _build_mocked_graph("billing", quality_approved=True)
        result = graph.invoke({**INITIAL_STATE, "ticket_text": "Billing issue"})
        assert result["quality_approved"] is True

    def test_rejected_pauses_at_quality_gate(self):
        checkpointer = MemorySaver()
        graph = _build_mocked_graph("billing", quality_approved=False, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-qa-interrupt"}}

        graph.invoke({**INITIAL_STATE, "ticket_text": "Billing issue"}, config=config)

        state = graph.get_state(config)
        assert any(
            hasattr(t, "interrupts") and t.interrupts for t in state.tasks
        )

    def test_approve_after_quality_interrupt(self):
        checkpointer = MemorySaver()
        graph = _build_mocked_graph("billing", quality_approved=False, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-qa-approve"}}

        graph.invoke({**INITIAL_STATE, "ticket_text": "Billing issue"}, config=config)

        result = graph.invoke(Command(resume="approve"), config=config)
        assert result["quality_approved"] is True

    def test_revise_after_quality_interrupt(self):
        checkpointer = MemorySaver()
        graph = _build_mocked_graph("billing", quality_approved=False, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-qa-revise"}}

        graph.invoke({**INITIAL_STATE, "ticket_text": "Billing issue"}, config=config)

        result = graph.invoke(Command(resume="Here is the corrected response."), config=config)
        assert result["quality_approved"] is True
        assert result["response"] == "Here is the corrected response."
