from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class TicketState(TypedDict):
    """Shared state that flows through every node in the ticket processing graph."""

    messages: Annotated[list, add_messages]
    ticket_text: str
    category: str
    confidence: float
    reasoning: str
    response: str


class TicketClassification(BaseModel):
    """Schema the classifier LLM must conform to via structured output."""

    category: Literal["billing", "technical", "general"]
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of why this category was chosen"
    )
