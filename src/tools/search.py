from langchain_core.tools import tool

from src.vector_store import get_retriever


def _format_results(docs) -> str:
    """Format retrieved documents with metadata for the LLM."""
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        directory = doc.metadata.get("directory", "unknown")
        parts.append(
            f"[{i}] (source: {directory}/{source})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def create_search_billing_docs(retriever=None):
    """Create a search tool for billing documentation."""
    _retriever = retriever or get_retriever("billing_docs")

    @tool
    def search_billing_docs(query: str) -> str:
        """Search billing documentation for information about refunds, invoices,
        subscriptions, payment methods, and billing policies. Use this tool when
        the customer has a billing-related question."""
        docs = _retriever.invoke(query)
        return _format_results(docs)

    return search_billing_docs


def create_search_technical_docs(retriever=None):
    """Create a search tool for technical documentation."""
    _retriever = retriever or get_retriever("technical_docs")

    @tool
    def search_technical_docs(query: str) -> str:
        """Search technical documentation for error codes, troubleshooting steps,
        API reference, and system requirements. Use this tool when the customer
        reports a technical issue or error."""
        docs = _retriever.invoke(query)
        return _format_results(docs)

    return search_technical_docs


def create_search_general_docs(retriever=None):
    """Create a search tool for general company documentation."""
    _retriever = retriever or get_retriever("general_docs")

    @tool
    def search_general_docs(query: str) -> str:
        """Search company documentation for business hours, office locations,
        contact information, product features, and company policies. Use this
        tool when the customer asks general questions about the company."""
        docs = _retriever.invoke(query)
        return _format_results(docs)

    return search_general_docs
