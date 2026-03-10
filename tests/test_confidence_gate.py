from unittest.mock import patch

from src.confidence_gate import CONFIDENCE_THRESHOLD, confidence_gate


class TestConfidenceGate:
    def test_high_confidence_passes_through(self, sample_state):
        sample_state["confidence"] = 0.9
        sample_state["category"] = "billing"
        result = confidence_gate(sample_state)
        assert result == {}

    def test_exactly_at_threshold_passes_through(self, sample_state):
        sample_state["confidence"] = CONFIDENCE_THRESHOLD
        sample_state["category"] = "billing"
        result = confidence_gate(sample_state)
        assert result == {}

    def test_low_confidence_with_override(self, sample_state):
        sample_state["confidence"] = 0.4
        sample_state["category"] = "general"
        sample_state["reasoning"] = "Unclear ticket"
        with patch("src.confidence_gate.interrupt", return_value="billing"):
            result = confidence_gate(sample_state)
        assert result["category"] == "billing"
        assert "billing" in result["messages"][0]["content"]

    def test_low_confidence_with_accept(self, sample_state):
        sample_state["confidence"] = 0.5
        sample_state["category"] = "technical"
        with patch("src.confidence_gate.interrupt", return_value="technical"):
            result = confidence_gate(sample_state)
        assert result["category"] == "technical"

    def test_low_confidence_with_invalid_input(self, sample_state):
        sample_state["confidence"] = 0.3
        sample_state["category"] = "billing"
        with patch("src.confidence_gate.interrupt", return_value="invalid"):
            result = confidence_gate(sample_state)
        assert result == {}

    def test_interrupt_payload_contains_info(self, sample_state):
        sample_state["confidence"] = 0.4
        sample_state["category"] = "general"
        sample_state["reasoning"] = "Ambiguous query"
        captured_payload = {}

        def fake_interrupt(payload):
            captured_payload.update(payload)
            return "general"

        with patch("src.confidence_gate.interrupt", side_effect=fake_interrupt):
            confidence_gate(sample_state)
        assert "0.40" in captured_payload["message"]
        assert captured_payload["current_category"] == "general"
        assert captured_payload["reasoning"] == "Ambiguous query"
        assert set(captured_payload["options"]) == {"billing", "technical", "general"}
