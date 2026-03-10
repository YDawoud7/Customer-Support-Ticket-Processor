from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agents import create_billing_agent, create_general_agent, create_technical_agent


def _mock_retriever(docs: list[str]):
    """Create a mock retriever that returns the given document texts."""
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content=text) for text in docs
    ]
    return retriever


def _mock_llm_response(content: str):
    """Patch ChatAnthropic to return a fixed AIMessage."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content=content)
    return mock_model


class TestBillingAgent:
    def test_returns_response_and_docs(self, sample_state):
        retriever = _mock_retriever(["Refund policy: 30 days."])
        mock_model = _mock_llm_response("We'll process your refund within 5 days.")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_billing_agent(retriever=retriever)
            result = agent(sample_state)
        assert result["response"] == "We'll process your refund within 5 days."
        assert len(result["retrieved_docs"]) == 1
        assert len(result["messages"]) == 1

    def test_calls_retriever_with_ticket_text(self, sample_state):
        retriever = _mock_retriever(["Some doc"])
        mock_model = _mock_llm_response("Response")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_billing_agent(retriever=retriever)
            agent(sample_state)
        retriever.invoke.assert_called_once_with(sample_state["ticket_text"])


class TestTechnicalAgent:
    def test_returns_response_and_docs(self, sample_state):
        retriever = _mock_retriever(["Error 0x8007: Export failure."])
        mock_model = _mock_llm_response("Clear your export cache to resolve this.")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_technical_agent(retriever=retriever)
            sample_state["ticket_text"] = "App crashes with 0x8007"
            result = agent(sample_state)
        assert result["response"] == "Clear your export cache to resolve this."
        assert len(result["retrieved_docs"]) == 1

    def test_calls_retriever_with_ticket_text(self, sample_state):
        retriever = _mock_retriever(["Some doc"])
        mock_model = _mock_llm_response("Response")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_technical_agent(retriever=retriever)
            agent(sample_state)
        retriever.invoke.assert_called_once_with(sample_state["ticket_text"])


class TestGeneralAgent:
    def test_returns_response_no_docs(self, sample_state):
        mock_model = _mock_llm_response("Our hours are 9am-5pm Monday to Friday.")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_general_agent()
            sample_state["ticket_text"] = "What are your hours?"
            result = agent(sample_state)
        assert result["response"] == "Our hours are 9am-5pm Monday to Friday."
        assert result["retrieved_docs"] == []
        assert len(result["messages"]) == 1
