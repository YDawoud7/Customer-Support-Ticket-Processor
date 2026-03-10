from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agents import (
    _run_tool_calling_loop,
    create_billing_agent,
    create_general_agent,
    create_technical_agent,
)


def _mock_model_direct_response(content: str):
    """Create a mock model that returns a text response (no tool calls)."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=content, tool_calls=[])
    mock.bind_tools.return_value = mock
    return mock


def _mock_model_with_tool_call(tool_name: str, tool_args: dict, tool_result: str, final_response: str):
    """Create a mock model that makes one tool call then gives a final response.

    First invoke returns an AIMessage with a tool call.
    Second invoke returns a plain text AIMessage.
    """
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": tool_name, "args": tool_args}],
    )
    final = AIMessage(content=final_response, tool_calls=[])

    mock = MagicMock()
    mock.invoke.side_effect = [tool_call_response, final]
    mock.bind_tools.return_value = mock
    return mock


def _mock_tool(name: str, result: str):
    """Create a mock tool."""
    tool = MagicMock()
    tool.name = name
    tool.invoke.return_value = result
    return tool


class TestToolCallingLoop:
    def test_direct_response_no_tools(self):
        model = MagicMock()
        model.invoke.return_value = AIMessage(content="Hello", tool_calls=[])
        response, docs = _run_tool_calling_loop(model, [], [])
        assert response.content == "Hello"
        assert docs == []

    def test_one_tool_call_round(self):
        tool = _mock_tool("search_billing_docs", "Refund policy: 30 days")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "search_billing_docs", "args": {"query": "refund"}}],
        )
        final_msg = AIMessage(content="Based on our policy...", tool_calls=[])
        model = MagicMock()
        model.invoke.side_effect = [tool_call_msg, final_msg]

        response, docs = _run_tool_calling_loop(model, [tool], [])
        assert response.content == "Based on our policy..."
        assert len(docs) == 1
        tool.invoke.assert_called_once_with({"query": "refund"})


class TestBillingAgent:
    def test_returns_response(self, sample_state):
        mock_model = _mock_model_direct_response("We'll process your refund.")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_billing_agent(tools=[])
            result = agent(sample_state)
        assert result["response"] == "We'll process your refund."
        assert len(result["messages"]) == 1

    def test_with_tool_call(self, sample_state):
        tool = _mock_tool("search_billing_docs", "30-day refund window")
        mock_model = _mock_model_with_tool_call(
            "search_billing_docs", {"query": "refund"}, "30-day refund", "Your refund is eligible."
        )
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_billing_agent(tools=[tool])
            result = agent(sample_state)
        assert result["response"] == "Your refund is eligible."
        assert len(result["retrieved_docs"]) == 1


class TestTechnicalAgent:
    def test_returns_response(self, sample_state):
        mock_model = _mock_model_direct_response("Clear your export cache.")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_technical_agent(tools=[])
            sample_state["ticket_text"] = "Error 0x8007"
            result = agent(sample_state)
        assert result["response"] == "Clear your export cache."

    def test_with_tool_call(self, sample_state):
        tool = _mock_tool("search_technical_docs", "Error 0x8007: export failure")
        mock_model = _mock_model_with_tool_call(
            "search_technical_docs", {"query": "0x8007"}, "export failure", "Clear the cache."
        )
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_technical_agent(tools=[tool])
            result = agent(sample_state)
        assert result["response"] == "Clear the cache."


class TestGeneralAgent:
    def test_returns_response(self, sample_state):
        mock_model = _mock_model_direct_response("Hours are 9am-5pm.")
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_general_agent(tools=[])
            sample_state["ticket_text"] = "What are your hours?"
            result = agent(sample_state)
        assert result["response"] == "Hours are 9am-5pm."

    def test_with_search_tool(self, sample_state):
        tool = _mock_tool("search_general_docs", "Hours: Mon-Fri 9am-5pm")
        mock_model = _mock_model_with_tool_call(
            "search_general_docs", {"query": "hours"}, "Mon-Fri 9am-5pm", "We're open Mon-Fri, 9am-5pm."
        )
        with patch("src.agents.ChatAnthropic", return_value=mock_model):
            agent = create_general_agent(tools=[tool])
            result = agent(sample_state)
        assert result["response"] == "We're open Mon-Fri, 9am-5pm."
        assert len(result["retrieved_docs"]) == 1
