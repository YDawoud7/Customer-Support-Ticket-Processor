from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph

SAMPLE_TICKETS = [
    "I was charged twice on my last invoice and need a refund for order #12345.",
    "My application keeps crashing with error code 0x8007 when I try to export to PDF.",
    "What are your business hours and do you have a store in Chicago?",
]


def main():
    app = build_graph()

    for ticket in SAMPLE_TICKETS:
        print(f"\n{'=' * 60}")
        print(f"TICKET: {ticket}")
        print("=" * 60)

        result = app.invoke(
            {
                "messages": [],
                "ticket_text": ticket,
                "category": "",
                "confidence": 0.0,
                "reasoning": "",
                "response": "",
            }
        )

        print(f"Category:   {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning:  {result['reasoning']}")
        print(f"Response:   {result['response']}")


if __name__ == "__main__":
    main()
