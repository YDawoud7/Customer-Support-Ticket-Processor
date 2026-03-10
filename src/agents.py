from src.state import TicketState


def billing_agent(state: TicketState) -> dict:
    return {
        "response": (
            "[Billing Agent] Thank you for your billing inquiry. "
            "We've reviewed your account and will process your request shortly."
        ),
        "messages": [
            {"role": "assistant", "content": "Billing agent processed the ticket."}
        ],
    }


def technical_agent(state: TicketState) -> dict:
    return {
        "response": (
            "[Technical Agent] Thank you for reporting this technical issue. "
            "Our engineering team has been notified and will investigate."
        ),
        "messages": [
            {"role": "assistant", "content": "Technical agent processed the ticket."}
        ],
    }


def general_agent(state: TicketState) -> dict:
    return {
        "response": (
            "[General Agent] Thank you for reaching out. "
            "A support representative will follow up with you shortly."
        ),
        "messages": [
            {"role": "assistant", "content": "General agent processed the ticket."}
        ],
    }
