from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph import build_graph, route_by_category
from src.state import QualityAssessment, TicketClassification


def _mock_all_llms(category: str, confidence: float = 0.9):
    """Context manager that mocks all LLM and retriever calls in the graph.

    Patches:
    - ChatAnthropic in classifier (structured output → TicketClassification)
    - ChatAnthropic in agents (invoke → AIMessage)
    - ChatAnthropic in quality_check (structured output → QualityAssessment)
    - get_retriever in graph (returns mock retriever)
    """
    # Classifier mock: with_structured_output().invoke() -> TicketClassification
    mock_classifier_structured = MagicMock()
    mock_classifier_structured.invoke.return_value = TicketClassification(
        category=category,
        confidence=confidence,
        reasoning=f"Mock classified as {category}",
    )

    # Quality check mock: with_structured_output().invoke() -> QualityAssessment
    mock_quality_structured = MagicMock()
    mock_quality_structured.invoke.return_value = QualityAssessment(
        approved=True, feedback="Mock approved."
    )

    # Agent mock: invoke() -> AIMessage
    mock_agent_model = MagicMock()
    mock_agent_model.invoke.return_value = AIMessage(
        content=f"Mock {category} agent response."
    )

    call_count = {"n": 0}

    def chat_anthropic_factory(*args, **kwargs):
        """Return different mocks depending on construction order.

        Order: classifier, billing/technical/general agent, quality_check.
        The classifier and quality_check use with_structured_output;
        agents use invoke directly.
        """
        mock = MagicMock()
        call_count["n"] += 1
        n = call_count["n"]
        if n == 1:
            # Classifier
            mock.with_structured_output.return_value = mock_classifier_structured
        elif n <= 4:
            # Agents (billing, technical, general)
            mock.invoke.return_value = AIMessage(
                content=f"Mock {category} agent response."
            )
        else:
            # Quality check
            mock.with_structured_output.return_value = mock_quality_structured
        return mock

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    patches = [
        patch("src.classifier.ChatAnthropic", side_effect=chat_anthropic_factory),
        patch("src.agents.ChatAnthropic", side_effect=chat_anthropic_factory),
        patch("src.quality_check.ChatAnthropic", side_effect=chat_anthropic_factory),
        patch("src.graph.get_retriever", return_value=mock_retriever),
    ]

    class PatchContext:
        def __enter__(self):
            for p in patches:
                p.__enter__()
            return self

        def __exit__(self, *args):
            for p in reversed(patches):
                p.__exit__(*args)

    return PatchContext()


def _build_mocked_graph(category: str, confidence: float = 0.9, checkpointer=None):
    with _mock_all_llms(category, confidence):
        return build_graph(checkpointer=checkpointer)


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

    def test_quality_check_runs_after_specialist(self):
        graph = _build_mocked_graph("billing")
        result = graph.invoke({**INITIAL_STATE, "ticket_text": "Billing issue"})
        assert "quality_feedback" in result
        assert result["quality_approved"] is True


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
