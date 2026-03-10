from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import TicketState

BILLING_SYSTEM_PROMPT = (
    "You are a billing support specialist. Use the provided documentation to help "
    "the customer with their billing issue. Be empathetic, reference specific "
    "policies when applicable, and provide clear actionable next steps.\n\n"
    "Documentation:\n{context}"
)

TECHNICAL_SYSTEM_PROMPT = (
    "You are a technical support specialist. Use the provided documentation to "
    "diagnose and resolve the customer's issue. Provide clear step-by-step "
    "instructions and reference specific error codes or procedures when applicable.\n\n"
    "Documentation:\n{context}"
)

GENERAL_SYSTEM_PROMPT = (
    "You are a friendly customer support representative. Provide helpful, "
    "accurate responses to the customer's inquiry. If you don't have specific "
    "information to answer their question, let them know and offer to connect "
    "them with a specialist who can help."
)


def create_billing_agent(model_name: str = "claude-sonnet-4-20250514", retriever=None):
    """Factory that returns a billing agent node with RAG capabilities."""
    model = ChatAnthropic(model=model_name)

    def billing_agent(state: TicketState) -> dict:
        docs = retriever.invoke(state["ticket_text"]) if retriever else []
        context = "\n\n".join(doc.page_content for doc in docs)
        doc_sources = [doc.page_content[:100] for doc in docs]

        response = model.invoke(
            [
                SystemMessage(content=BILLING_SYSTEM_PROMPT.format(context=context)),
                HumanMessage(content=state["ticket_text"]),
            ]
        )
        return {
            "response": response.content,
            "retrieved_docs": doc_sources,
            "messages": [
                {"role": "assistant", "content": "Billing agent processed the ticket."}
            ],
        }

    return billing_agent


def create_technical_agent(model_name: str = "claude-sonnet-4-20250514", retriever=None):
    """Factory that returns a technical agent node with RAG capabilities."""
    model = ChatAnthropic(model=model_name)

    def technical_agent(state: TicketState) -> dict:
        docs = retriever.invoke(state["ticket_text"]) if retriever else []
        context = "\n\n".join(doc.page_content for doc in docs)
        doc_sources = [doc.page_content[:100] for doc in docs]

        response = model.invoke(
            [
                SystemMessage(content=TECHNICAL_SYSTEM_PROMPT.format(context=context)),
                HumanMessage(content=state["ticket_text"]),
            ]
        )
        return {
            "response": response.content,
            "retrieved_docs": doc_sources,
            "messages": [
                {
                    "role": "assistant",
                    "content": "Technical agent processed the ticket.",
                }
            ],
        }

    return technical_agent


def create_general_agent(model_name: str = "claude-sonnet-4-20250514"):
    """Factory that returns a general agent node (no RAG, just conversational)."""
    model = ChatAnthropic(model=model_name)

    def general_agent(state: TicketState) -> dict:
        response = model.invoke(
            [
                SystemMessage(content=GENERAL_SYSTEM_PROMPT),
                HumanMessage(content=state["ticket_text"]),
            ]
        )
        return {
            "response": response.content,
            "retrieved_docs": [],
            "messages": [
                {"role": "assistant", "content": "General agent processed the ticket."}
            ],
        }

    return general_agent
