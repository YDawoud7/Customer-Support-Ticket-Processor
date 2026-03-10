from src.vector_store import build_vector_store, get_retriever, load_and_split_documents


class TestLoadAndSplitDocuments:
    def test_loads_billing_docs(self):
        chunks = load_and_split_documents("data/billing_docs")
        assert len(chunks) > 0
        assert all(c.page_content for c in chunks)

    def test_loads_technical_docs(self):
        chunks = load_and_split_documents("data/technical_docs")
        assert len(chunks) > 0

    def test_chunks_have_source_metadata(self):
        chunks = load_and_split_documents("data/billing_docs")
        sources = {c.metadata["source"] for c in chunks}
        assert "refund_policy.md" in sources

    def test_chunks_are_within_size_limit(self):
        chunks = load_and_split_documents("data/billing_docs")
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # 500 + some tolerance


class TestBuildAndRetrieve:
    def test_build_and_retrieve_billing(self, tmp_path):
        persist_dir = str(tmp_path / "chroma_test")
        build_vector_store("billing_docs", "data/billing_docs", persist_dir)
        retriever = get_retriever("billing_docs", persist_dir, k=2)
        results = retriever.invoke("duplicate charge refund")
        assert len(results) > 0
        assert any("refund" in r.page_content.lower() for r in results)

    def test_build_and_retrieve_technical(self, tmp_path):
        persist_dir = str(tmp_path / "chroma_test")
        build_vector_store("technical_docs", "data/technical_docs", persist_dir)
        retriever = get_retriever("technical_docs", persist_dir, k=2)
        results = retriever.invoke("error code 0x8007 PDF export")
        assert len(results) > 0
        assert any("0x8007" in r.page_content for r in results)

    def test_missing_store_raises_error(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError, match="Vector store not found"):
            get_retriever("nonexistent", str(tmp_path / "nope"))
