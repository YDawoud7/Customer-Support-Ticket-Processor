from unittest.mock import patch

from src.quality_gate import quality_gate


class TestQualityGate:
    def test_approved_passes_through(self, sample_state):
        sample_state["quality_approved"] = True
        result = quality_gate(sample_state)
        assert result == {}

    def test_flagged_with_human_approve(self, sample_state):
        sample_state["quality_approved"] = False
        sample_state["response"] = "Draft response"
        sample_state["quality_feedback"] = "Tone too casual"
        with patch("src.quality_gate.interrupt", return_value="approve"):
            result = quality_gate(sample_state)
        assert result["quality_approved"] is True
        assert "response" not in result  # Original response kept

    def test_flagged_with_human_revision(self, sample_state):
        sample_state["quality_approved"] = False
        sample_state["response"] = "Bad draft"
        sample_state["quality_feedback"] = "Incomplete"
        revised = "Here is the corrected, complete response."
        with patch("src.quality_gate.interrupt", return_value=revised):
            result = quality_gate(sample_state)
        assert result["quality_approved"] is True
        assert result["response"] == revised

    def test_interrupt_payload_has_context(self, sample_state):
        sample_state["quality_approved"] = False
        sample_state["response"] = "Draft"
        sample_state["quality_feedback"] = "Needs work"
        sample_state["category"] = "billing"
        sample_state["ticket_text"] = "Refund please"
        captured = {}

        def fake_interrupt(payload):
            captured.update(payload)
            return "approve"

        with patch("src.quality_gate.interrupt", side_effect=fake_interrupt):
            quality_gate(sample_state)
        assert captured["draft_response"] == "Draft"
        assert captured["quality_feedback"] == "Needs work"
        assert captured["category"] == "billing"
        assert captured["ticket_text"] == "Refund please"
