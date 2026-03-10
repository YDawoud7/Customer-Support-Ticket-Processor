from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.state import TicketState

MAX_TOOL_ROUNDS = 5

BILLING_SYSTEM_PROMPT = (
    "You are a billing support specialist. You have access to tools that let you "
    "search billing documentation and perform calculations. Always search the "
    "documentation before answering to ensure accuracy. Use the calculator for "
    "any refund amounts, pro-rated charges, or cost comparisons.\n\n"
    "Be empathetic, reference specific policies when applicable, and provide "
    "clear actionable next steps."
)

TECHNICAL_SYSTEM_PROMPT = (
    "You are a technical support specialist. You have access to tools that let you "
    "search technical documentation and analyze code snippets. Always search the "
    "documentation for error codes and troubleshooting steps before answering.\n\n"
    "Provide clear step-by-step instructions and reference specific error codes "
    "or procedures from the documentation."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a friendly customer support representative. You have access to a tool "
    "that lets you search company documentation for business hours, office locations, "
    "product information, and company policies. Always search before answering to "
    "provide accurate information.\n\n"
    "If you cannot find the answer in the documentation, let the customer know "
    "and offer to connect them with a specialist."
)


def _run_tool_calling_loop(model, tools, messages, max_rounds=MAX_TOOL_ROUNDS):
    """Run the tool-calling loop: LLM calls tools, we execute them, repeat.

    Returns the final AIMessage (text response) and a list of document
    excerpts retrieved during tool calls.
    """
    tools_by_name = {t.name: t for t in tools}
    retrieved_docs = []

    for _ in range(max_rounds):
        response: AIMessage = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response, retrieved_docs

        for tool_call in response.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])

            if "search" in tool_call["name"]:
                retrieved_docs.append(
                    f"[{tool_call['name']}] query='{tool_call['args'].get('query', '')}'"
                )

            messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    return messages[-1], retrieved_docs


def create_billing_agent(model_name: str = "claude-sonnet-4-20250514", tools=None):
    """Factory that returns a billing agent with tool-calling capabilities."""
    model = ChatAnthropic(model=model_name)
    agent_tools = tools or []
    model_with_tools = model.bind_tools(agent_tools) if agent_tools else model

    def billing_agent(state: TicketState) -> dict:
        messages = [
            SystemMessage(content=BILLING_SYSTEM_PROMPT),
            HumanMessage(content=state["ticket_text"]),
        ]
        response, retrieved_docs = _run_tool_calling_loop(
            model_with_tools, agent_tools, messages
        )
        return {
            "response": response.content,
            "retrieved_docs": retrieved_docs,
            "messages": [
                {"role": "assistant", "content": "Billing agent processed the ticket."}
            ],
        }

    return billing_agent


def create_technical_agent(model_name: str = "claude-sonnet-4-20250514", tools=None):
    """Factory that returns a technical agent with tool-calling capabilities."""
    model = ChatAnthropic(model=model_name)
    agent_tools = tools or []
    model_with_tools = model.bind_tools(agent_tools) if agent_tools else model

    def technical_agent(state: TicketState) -> dict:
        messages = [
            SystemMessage(content=TECHNICAL_SYSTEM_PROMPT),
            HumanMessage(content=state["ticket_text"]),
        ]
        response, retrieved_docs = _run_tool_calling_loop(
            model_with_tools, agent_tools, messages
        )
        return {
            "response": response.content,
            "retrieved_docs": retrieved_docs,
            "messages": [
                {
                    "role": "assistant",
                    "content": "Technical agent processed the ticket.",
                }
            ],
        }

    return technical_agent


def create_general_agent(model_name: str = "claude-sonnet-4-20250514", tools=None):
    """Factory that returns a general agent with tool-calling capabilities."""
    model = ChatAnthropic(model=model_name)
    agent_tools = tools or []
    model_with_tools = model.bind_tools(agent_tools) if agent_tools else model

    def general_agent(state: TicketState) -> dict:
        messages = [
            SystemMessage(content=GENERAL_SYSTEM_PROMPT),
            HumanMessage(content=state["ticket_text"]),
        ]
        response, retrieved_docs = _run_tool_calling_loop(
            model_with_tools, agent_tools, messages
        )
        return {
            "response": response.content,
            "retrieved_docs": retrieved_docs,
            "messages": [
                {"role": "assistant", "content": "General agent processed the ticket."}
            ],
        }

    return general_agent
