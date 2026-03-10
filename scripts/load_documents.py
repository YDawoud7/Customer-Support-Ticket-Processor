"""One-time script to initialize ChromaDB vector stores from the RAG documents."""

from src.vector_store import build_vector_store

print("Building billing docs vector store...")
build_vector_store("billing_docs", "data/billing_docs")
print("Building technical docs vector store...")
build_vector_store("technical_docs", "data/technical_docs")
print("Building general docs vector store...")
build_vector_store("general_docs", "data/general_docs")
print("Done. Vector stores saved to ./chroma_db/")
