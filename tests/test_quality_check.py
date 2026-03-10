from unittest.mock import MagicMock, patch

from src.quality_check import create_quality_check
from src.state import QualityAssessment


def _build_quality_check_with_mock(assessment: QualityAssessment):
    """Create a quality check node with a mocked LLM."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = assessment

    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    with patch("src.quality_check.ChatAnthropic", return_value=mock_model):
        node = create_quality_check()

    return node


class TestQualityCheckNode:
    def test_approved_response(self, sample_state):
        sample_state["response"] = "We'll process your refund within 5 days."
        sample_state["category"] = "billing"
        assessment = QualityAssessment(
            approved=True, feedback="Professional tone, addresses the issue clearly."
        )
        node = _build_quality_check_with_mock(assessment)
        result = node(sample_state)
        assert result["quality_approved"] is True
        assert "Professional" in result["quality_feedback"]
        assert "Approved" in result["messages"][0]["content"]

    def test_rejected_response(self, sample_state):
        sample_state["response"] = "idk try again lol"
        sample_state["category"] = "technical"
        assessment = QualityAssessment(
            approved=False,
            feedback="Tone is too casual and does not provide actionable steps.",
        )
        node = _build_quality_check_with_mock(assessment)
        result = node(sample_state)
        assert result["quality_approved"] is False
        assert "casual" in result["quality_feedback"]
        assert "revision" in result["messages"][0]["content"]

    def test_appends_message(self, sample_state):
        sample_state["response"] = "Some response"
        sample_state["category"] = "general"
        assessment = QualityAssessment(approved=True, feedback="Good.")
        node = _build_quality_check_with_mock(assessment)
        result = node(sample_state)
        assert len(result["messages"]) == 1
