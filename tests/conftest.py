import pytest

from src.state import TicketClassification


@pytest.fixture
def sample_state():
    return {
        "messages": [],
        "ticket_text": "I was charged twice on my last invoice.",
        "category": "",
        "confidence": 0.0,
        "reasoning": "",
        "response": "",
        "retrieved_docs": [],
        "quality_approved": False,
        "quality_feedback": "",
    }


@pytest.fixture
def mock_classification():
    return TicketClassification(
        category="billing",
        confidence=0.95,
        reasoning="Customer mentions being charged twice on an invoice.",
    )
