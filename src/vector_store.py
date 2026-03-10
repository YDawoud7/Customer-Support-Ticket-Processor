from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_PERSIST_DIR = "./chroma_db"


def get_embeddings():
    """Return a local embedding model. No API key needed."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_and_split_documents(docs_dir: str) -> list[Document]:
    """Load markdown files from a directory and split into chunks."""
    docs_path = Path(docs_dir)
    documents = []
    for md_file in sorted(docs_path.glob("*.md")):
        text = md_file.read_text()
        documents.append(
            Document(page_content=text, metadata={"source": md_file.name})
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def build_vector_store(
    collection_name: str,
    docs_dir: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """Load documents, embed them, and persist to a ChromaDB collection."""
    chunks = load_and_split_documents(docs_dir)
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return vector_store


def get_retriever(
    collection_name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    k: int = 3,
):
    """Open an existing ChromaDB collection and return a retriever.

    Raises FileNotFoundError if the vector store has not been initialized.
    Run `python scripts/load_documents.py` to initialize.
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{persist_dir}'. "
            "Run: python scripts/load_documents.py"
        )

    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
