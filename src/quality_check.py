from langchain_core.messages import HumanMessage, SystemMessage

from src.models import create_chat_model
from src.state import QualityAssessment, TicketState

QUALITY_CHECK_SYSTEM_PROMPT = (
    "You are a quality assurance reviewer for customer support responses. "
    "Evaluate the draft response against the original ticket and assess:\n"
    "- **Tone**: Is the response professional, empathetic, and appropriate?\n"
    "- **Completeness**: Does it address all parts of the customer's inquiry?\n"
    "- **Accuracy**: Is the information consistent with the provided context?\n"
    "- **Clarity**: Are next steps clear and actionable?\n\n"
    "Approve the response if it meets all criteria. Reject with specific feedback "
    "if it falls short in any area."
)


def create_quality_check(model_name: str = "claude-sonnet-4-20250514"):
    """Factory that returns a quality check node function."""
    model = create_chat_model(model_name)
    if model_name.startswith("deepseek"):
        # The default method 'json-schema' doesn't work with Deepseek, so we switch to function calling
        structured_model = model.with_structured_output(QualityAssessment, method='function_calling')
    else:
        structured_model = model.with_structured_output(QualityAssessment)

    def quality_check_node(state: TicketState) -> dict:
        review_prompt = (
            f"Original ticket: {state['ticket_text']}\n"
            f"Category: {state['category']}\n"
            f"Draft response: {state['response']}"
        )

        result: QualityAssessment = structured_model.invoke(
            [
                SystemMessage(content=QUALITY_CHECK_SYSTEM_PROMPT),
                HumanMessage(content=review_prompt),
            ]
        )
        return {
            "quality_approved": result.approved,
            "quality_feedback": result.feedback,
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Quality check: {'Approved' if result.approved else 'Needs revision'}"
                    ),
                }
            ],
        }

    return quality_check_node
