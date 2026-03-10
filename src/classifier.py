from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_chat_model
from src.state import TicketClassification, TicketState

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a customer support ticket classifier. "
    "Classify the ticket into exactly one category:\n"
    "- billing: payment issues, invoices, charges, refunds, subscriptions\n"
    "- technical: bugs, errors, crashes, performance, integrations\n"
    "- general: store hours, company info, product questions, anything else\n\n"
    "Provide your confidence score and brief reasoning."
)


def create_classifier(model_name: str = "claude-sonnet-4-20250514"):
    """Factory that returns a classifier node function.

    Uses a factory so the LLM is only instantiated when the graph is built
    (after load_dotenv), not at import time.
    """
    model = create_chat_model(model_name)
    if model_name.startswith('deepseek'):
        # The default method 'json-schema' doesn't work with Deepseek, so we switch to function calling
        structured_model = model.with_structured_output(TicketClassification, method='function_calling')
    else:
        structured_model = model.with_structured_output(TicketClassification)

    def classifier_node(state: TicketState) -> dict:
        result: TicketClassification = structured_model.invoke(
            [
                SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=state["ticket_text"]),
            ]
        )
        return {
            "category": result.category,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Classified as {result.category} "
                        f"(confidence: {result.confidence:.2f})"
                    ),
                }
            ],
        }

    return classifier_node
