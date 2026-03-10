import pytest
from pydantic import ValidationError

from src.state import QualityAssessment, TicketClassification


class TestTicketClassification:
    def test_valid_billing(self):
        result = TicketClassification(
            category="billing", confidence=0.9, reasoning="Invoice issue"
        )
        assert result.category == "billing"

    def test_valid_technical(self):
        result = TicketClassification(
            category="technical", confidence=0.8, reasoning="App crash"
        )
        assert result.category == "technical"

    def test_valid_general(self):
        result = TicketClassification(
            category="general", confidence=0.7, reasoning="Store hours question"
        )
        assert result.category == "general"

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            TicketClassification(
                category="shipping", confidence=0.9, reasoning="Package lost"
            )

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            TicketClassification(
                category="billing", confidence=-0.1, reasoning="Test"
            )

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            TicketClassification(
                category="billing", confidence=1.1, reasoning="Test"
            )

    def test_confidence_boundary_zero(self):
        result = TicketClassification(
            category="billing", confidence=0.0, reasoning="Low confidence"
        )
        assert result.confidence == 0.0

    def test_confidence_boundary_one(self):
        result = TicketClassification(
            category="billing", confidence=1.0, reasoning="High confidence"
        )
        assert result.confidence == 1.0


class TestQualityAssessment:
    def test_approved_response(self):
        result = QualityAssessment(approved=True, feedback="Response is thorough.")
        assert result.approved is True
        assert len(result.feedback) > 0

    def test_rejected_response(self):
        result = QualityAssessment(approved=False, feedback="Tone is too casual.")
        assert result.approved is False

    def test_requires_feedback(self):
        with pytest.raises(ValidationError):
            QualityAssessment(approved=True)
