import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from scripts.run_eval import compute_metrics, generate_report, load_eval_tickets, run_ticket
from src.graph import build_graph
from src.state import QualityAssessment, TicketClassification


EVAL_TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_tickets.json"


class TestLoadEvalTickets:
    def test_loads_all_tickets(self):
        tickets = load_eval_tickets()
        assert len(tickets) == 30

    def test_each_ticket_has_required_fields(self):
        tickets = load_eval_tickets()
        for ticket in tickets:
            assert "id" in ticket
            assert "ticket_text" in ticket
            assert "expected_category" in ticket

    def test_category_distribution(self):
        tickets = load_eval_tickets()
        categories = [t["expected_category"] for t in tickets]
        assert categories.count("billing") == 10
        assert categories.count("technical") == 10
        assert categories.count("general") == 10


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        results = [
            {"ticket_id": "b1", "expected": "billing", "predicted": "billing", "correct": True, "confidence": 0.95, "quality_approved": True, "latency": 2.0, "response_length": 100},
            {"ticket_id": "t1", "expected": "technical", "predicted": "technical", "correct": True, "confidence": 0.90, "quality_approved": True, "latency": 3.0, "response_length": 150},
        ]
        metrics = compute_metrics(results)
        assert metrics["accuracy"] == 1.0
        assert metrics["quality_pass_rate"] == 1.0
        assert metrics["avg_latency"] == 2.5
        assert metrics["avg_confidence"] == 0.925

    def test_partial_accuracy(self):
        results = [
            {"ticket_id": "b1", "expected": "billing", "predicted": "billing", "correct": True, "confidence": 0.9, "quality_approved": True, "latency": 2.0, "response_length": 100},
            {"ticket_id": "b2", "expected": "billing", "predicted": "general", "correct": False, "confidence": 0.5, "quality_approved": False, "latency": 1.0, "response_length": 50},
        ]
        metrics = compute_metrics(results)
        assert metrics["accuracy"] == 0.5
        assert metrics["quality_pass_rate"] == 0.5
        assert metrics["billing_accuracy"] == 0.5
        assert metrics["billing_correct"] == 1
        assert metrics["billing_total"] == 2

    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["accuracy"] == 0
        assert metrics["avg_latency"] == 0

    def test_per_category_breakdown(self):
        results = [
            {"ticket_id": "b1", "expected": "billing", "predicted": "billing", "correct": True, "confidence": 0.9, "quality_approved": True, "latency": 2.0, "response_length": 100},
            {"ticket_id": "t1", "expected": "technical", "predicted": "billing", "correct": False, "confidence": 0.6, "quality_approved": True, "latency": 3.0, "response_length": 100},
            {"ticket_id": "g1", "expected": "general", "predicted": "general", "correct": True, "confidence": 0.8, "quality_approved": True, "latency": 1.5, "response_length": 100},
        ]
        metrics = compute_metrics(results)
        assert metrics["billing_accuracy"] == 1.0
        assert metrics["technical_accuracy"] == 0.0
        assert metrics["general_accuracy"] == 1.0


class TestGenerateReport:
    def test_report_contains_summary_table(self, tmp_path):
        results = {
            "all_claude": [
                {"ticket_id": "b1", "expected": "billing", "predicted": "billing", "correct": True, "confidence": 0.95, "quality_approved": True, "latency": 2.0, "response_length": 100},
            ],
        }
        output = tmp_path / "report.md"
        report = generate_report(results, str(output))
        assert "# Evaluation Report" in report
        assert "## Summary" in report
        assert "Accuracy" in report
        assert "all_claude" in report

    def test_report_has_per_category_section(self, tmp_path):
        results = {
            "test_config": [
                {"ticket_id": "b1", "expected": "billing", "predicted": "billing", "correct": True, "confidence": 0.9, "quality_approved": True, "latency": 2.0, "response_length": 100},
            ],
        }
        output = tmp_path / "report.md"
        report = generate_report(results, str(output))
        assert "## Per-Category Breakdown" in report
        assert "### Billing" in report

    def test_report_writes_file(self, tmp_path):
        results = {
            "test_config": [
                {"ticket_id": "b1", "expected": "billing", "predicted": "billing", "correct": True, "confidence": 0.9, "quality_approved": True, "latency": 2.0, "response_length": 100},
            ],
        }
        output = tmp_path / "report.md"
        generate_report(results, str(output))
        assert output.exists()
        assert len(output.read_text()) > 0


def _build_mocked_graph_for_eval(category, confidence=0.9, quality_approved=True):
    """Build a mocked graph for eval testing (same pattern as test_graph.py)."""
    mock_classifier_structured = MagicMock()
    mock_classifier_structured.invoke.return_value = TicketClassification(
        category=category,
        confidence=confidence,
        reasoning=f"Mock classified as {category}",
    )

    mock_quality_structured = MagicMock()
    mock_quality_structured.invoke.return_value = QualityAssessment(
        approved=quality_approved,
        feedback="Mock approved." if quality_approved else "Mock rejected.",
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

    checkpointer = MemorySaver()
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


class TestAutoResumeInterrupts:
    def test_auto_resume_confidence_interrupt(self):
        graph = _build_mocked_graph_for_eval("general", confidence=0.3)
        ticket = {"id": "test-1", "ticket_text": "Unclear ticket", "expected_category": "general"}
        result = run_ticket(graph, ticket, "eval-test-confidence")
        assert result["category"] == "general"
        assert len(result["response"]) > 0

    def test_auto_resume_quality_interrupt(self):
        graph = _build_mocked_graph_for_eval("billing", quality_approved=False)
        ticket = {"id": "test-2", "ticket_text": "Billing issue", "expected_category": "billing"}
        result = run_ticket(graph, ticket, "eval-test-quality")
        assert result["quality_approved"] is True
