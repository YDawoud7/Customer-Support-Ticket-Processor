from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph import build_graph

SAMPLE_TICKETS = [
    "I was charged twice on my last invoice and need a refund for order #12345.",
    "My application keeps crashing with error code 0x8007 when I try to export to PDF.",
    "What are your business hours and do you have a store in Chicago?",
]

INITIAL_STATE = {
    "messages": [],
    "ticket_text": "",
    "category": "",
    "confidence": 0.0,
    "reasoning": "",
    "response": "",
    "retrieved_docs": [],
    "quality_approved": False,
    "quality_feedback": "",
}


def process_ticket(app, ticket: str, thread_id: str):
    """Process a single ticket, handling any interrupt for human review."""
    config = {"configurable": {"thread_id": thread_id}}
    state = {**INITIAL_STATE, "ticket_text": ticket}

    result = app.invoke(state, config=config)

    # Check if the graph paused at an interrupt
    graph_state = app.get_state(config)
    while graph_state.tasks and any(
        hasattr(t, "interrupts") and t.interrupts for t in graph_state.tasks
    ):
        interrupt_info = graph_state.tasks[0].interrupts[0].value
        print(f"\n  ** HUMAN REVIEW NEEDED **")
        print(f"  {interrupt_info['message']}")
        print(f"  Current category: {interrupt_info['current_category']}")
        print(f"  Reasoning: {interrupt_info['reasoning']}")

        choice = input(f"  Enter category {interrupt_info['options']}: ").strip()
        if choice not in ("billing", "technical", "general"):
            choice = interrupt_info["current_category"]
            print(f"  Invalid input, accepting: {choice}")

        result = app.invoke(Command(resume=choice), config=config)
        graph_state = app.get_state(config)

    return result


def main():
    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)

    for i, ticket in enumerate(SAMPLE_TICKETS):
        print(f"\n{'=' * 60}")
        print(f"TICKET: {ticket}")
        print("=" * 60)

        result = process_ticket(app, ticket, thread_id=f"ticket-{i}")

        print(f"Category:       {result['category']}")
        print(f"Confidence:     {result['confidence']:.2f}")
        print(f"Reasoning:      {result['reasoning']}")
        print(f"Response:       {result['response'][:200]}...")
        print(f"Quality:        {'Approved' if result['quality_approved'] else 'Needs revision'}")
        print(f"QA Feedback:    {result['quality_feedback'][:150]}...")
        if result["retrieved_docs"]:
            print(f"Retrieved Docs: {len(result['retrieved_docs'])} chunks used")


if __name__ == "__main__":
    main()
