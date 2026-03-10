from langgraph.types import interrupt

from src.state import TicketState

CONFIDENCE_THRESHOLD = 0.7


def confidence_gate(state: TicketState) -> dict:
    """Pause for human review if classifier confidence is below threshold.

    When confidence is high enough, this node passes through with no state
    changes. When low, it calls interrupt() which pauses the graph and
    returns control to the caller. The caller resumes with
    Command(resume="billing"|"technical"|"general") to continue.
    """
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        return {}

    user_decision = interrupt(
        {
            "message": (
                f"Low confidence classification ({state['confidence']:.2f}). "
                "Please review and confirm or override the category."
            ),
            "current_category": state["category"],
            "reasoning": state["reasoning"],
            "options": ["billing", "technical", "general"],
        }
    )

    if user_decision in ("billing", "technical", "general"):
        return {
            "category": user_decision,
            "messages": [
                {
                    "role": "user",
                    "content": f"Human review: category set to {user_decision}",
                }
            ],
        }
    return {}
