from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

# Ensure the package root is importable when running pytest from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rag_pipeline  # noqa: E402  (added after sys.path manipulation)
from pdf_processor import DocumentChunk, ProcessedDocument  # noqa: E402
from rag_pipeline import DocumentIngestionPipeline, RAGModule  # noqa: E402
from retriever import HybridRetriever, RetrievalResult  # noqa: E402


class FakeGenerator:
    """Lightweight stand-in for the dspy predictor."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, context: str, question: str):
        self.calls.append({"context": context, "question": question})
        return SimpleNamespace(
            answer="test-answer",
            reasoning="test-reasoning",
            sources="doc1",
        )


def fake_retriever(query: str, k: int = 5):
    return [
        RetrievalResult(content="First chunk", source="doc1.pdf", section="Intro"),
        RetrievalResult(content="Second chunk", source="doc2.pdf"),
    ]


def test_build_context_formats_sources():
    rag = RAGModule.__new__(RAGModule)

    context = rag._build_context(
        [
            RetrievalResult(content="A", source="doc1.pdf", section="Overview"),
            RetrievalResult(content="B", source="doc2.pdf"),
        ]
    )

    assert "[Source 1: doc1.pdf - Overview]" in context
    assert "[Source 2: doc2.pdf]" in context
    assert "---" in context


def test_forward_returns_response_with_retrieval_metadata():
    generator = FakeGenerator()
    rag = RAGModule.__new__(RAGModule)
    rag.retriever = fake_retriever
    rag.k = 2
    rag.generate = generator

    response = rag.forward("What is inside?")

    assert response.answer == "test-answer"
    assert response.reasoning == "test-reasoning"
    assert response.sources == "doc1"
    assert len(response.retrieved_chunks) == 2
    assert generator.calls[0]["question"] == "What is inside?"
    assert "[Source 1: doc1.pdf - Intro]" in generator.calls[0]["context"]


def test_ingest_routes_chunks_through_pipeline(monkeypatch: pytest.MonkeyPatch):
    class FakePDFProcessor:
        def __init__(self, enable_ocr: bool = True):
            self.enable_ocr = enable_ocr
            self.process_calls = []
            self.chunk_calls = []

        def process(self, source):
            self.process_calls.append(source)
            return ProcessedDocument(
                source=str(source),
                markdown="# Title\nContent",
                metadata={"title": "Doc"},
                pages=1,
                tables_count=0,
                ocr_used=False,
            )

        def chunk_document(self, doc, method: str = "semantic"):
            self.chunk_calls.append((doc.source, method))
            return [
                DocumentChunk(
                    content=doc.markdown,
                    source=doc.source,
                    section="Title",
                    metadata=doc.metadata.copy(),
                )
            ]

    class FakeEmbeddingPipeline:
        def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
            self.received = []
            self.embedding_model = embedding_model

        def process_and_store(self, chunks, batch_size: int = 32):
            self.received.append((chunks, batch_size))
            return len(chunks)

    fake_pdf = FakePDFProcessor(enable_ocr=False)
    fake_embedder = FakeEmbeddingPipeline()

    monkeypatch.setattr(rag_pipeline, "PDFProcessor", lambda enable_ocr=True: fake_pdf)
    monkeypatch.setattr(
        rag_pipeline, "EmbeddingPipeline", lambda embedding_model="all-MiniLM-L6-v2": fake_embedder
    )

    pipeline = DocumentIngestionPipeline(enable_ocr=False, enable_contextual=False)

    inserted = pipeline.ingest("sample.pdf", metadata={"category": "guide"})

    assert inserted == 1
    assert fake_pdf.process_calls == ["sample.pdf"]
    assert fake_pdf.chunk_calls == [("sample.pdf", "semantic")]
    assert fake_embedder.received
    stored_chunks, batch_size = fake_embedder.received[0]
    assert batch_size == 32
    assert stored_chunks[0].metadata["category"] == "guide"
    assert stored_chunks[0].section == "Title"


def test_rrf_prefers_documents_present_in_both_rankings():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.bm25_weight = 0.2
    retriever.vector_weight = 1.0

    bm25_results = [(0, 2.0), (1, 1.0)]
    vector_results = [(1, 0.9, {}), (2, 0.8, {})]

    fused = retriever._reciprocal_rank_fusion(
        bm25_results, vector_results, k_constant=1
    )

    assert fused[0][0] == 1  # Appears in both lists with strongest combined weight
    assert fused[0][1] > fused[1][1]
    assert {idx for idx, _ in fused} == {0, 1, 2}
