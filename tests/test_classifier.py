from unittest.mock import MagicMock, patch

from src.classifier import CLASSIFIER_SYSTEM_PROMPT, create_classifier
from src.state import TicketClassification


class TestClassifierNode:
    def _build_classifier_with_mock(self, classification: TicketClassification):
        """Create a classifier node with a mocked LLM that returns the given classification."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = classification

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with patch("src.classifier.ChatAnthropic", return_value=mock_model):
            node = create_classifier()

        return node

    def test_returns_correct_category(self, sample_state, mock_classification):
        node = self._build_classifier_with_mock(mock_classification)
        result = node(sample_state)
        assert result["category"] == "billing"

    def test_returns_confidence(self, sample_state, mock_classification):
        node = self._build_classifier_with_mock(mock_classification)
        result = node(sample_state)
        assert result["confidence"] == 0.95

    def test_returns_reasoning(self, sample_state, mock_classification):
        node = self._build_classifier_with_mock(mock_classification)
        result = node(sample_state)
        assert "charged twice" in result["reasoning"]

    def test_appends_message(self, sample_state, mock_classification):
        node = self._build_classifier_with_mock(mock_classification)
        result = node(sample_state)
        assert len(result["messages"]) == 1
        assert "billing" in result["messages"][0]["content"]

    def test_technical_classification(self, sample_state):
        classification = TicketClassification(
            category="technical",
            confidence=0.85,
            reasoning="Application crash with error code.",
        )
        node = self._build_classifier_with_mock(classification)
        sample_state["ticket_text"] = "App crashes with error 0x8007"
        result = node(sample_state)
        assert result["category"] == "technical"

    def test_general_classification(self, sample_state):
        classification = TicketClassification(
            category="general",
            confidence=0.75,
            reasoning="Asking about store hours.",
        )
        node = self._build_classifier_with_mock(classification)
        sample_state["ticket_text"] = "What are your business hours?"
        result = node(sample_state)
        assert result["category"] == "general"
