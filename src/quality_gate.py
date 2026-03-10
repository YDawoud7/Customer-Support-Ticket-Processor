from langgraph.types import interrupt

from src.state import TicketState


def quality_gate(state: TicketState) -> dict:
    """Pause for human review if the quality check flagged the response.

    When quality_approved is True, this node passes through. When False,
    it interrupts with the draft response and feedback for human review.
    The human can approve (resume with "approve") or provide a revised
    response (resume with the replacement text).
    """
    if state["quality_approved"]:
        return {}

    user_decision = interrupt(
        {
            "message": "Quality check flagged this response for review.",
            "draft_response": state["response"],
            "quality_feedback": state["quality_feedback"],
            "category": state["category"],
            "ticket_text": state["ticket_text"],
            "options": ["approve", "or provide a revised response"],
        }
    )

    if user_decision == "approve":
        return {
            "quality_approved": True,
            "messages": [
                {"role": "user", "content": "Human approved the flagged response."}
            ],
        }

    # User provided a revised response
    return {
        "response": user_decision,
        "quality_approved": True,
        "messages": [
            {"role": "user", "content": "Human provided a revised response."}
        ],
    }
