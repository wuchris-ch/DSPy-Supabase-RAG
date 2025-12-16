"""
DSPy Supabase RAG Pipeline
==========================

A production-ready RAG pipeline with:
- Docling for intelligent PDF parsing
- Supabase pgvector for vector storage
- DSPy for structured LLM programming
- RAGAS for evaluation

Quick Start:
    from rag_pipeline import RAGSystem
    
    rag = RAGSystem()
    rag.ingest("document.pdf")
    response = rag.query("What is the main topic?")
    print(response.answer)
"""

# Absolute imports so this module works even when the parent directory
# is added directly to sys.path (e.g., during pytest collection).
from pdf_processor import PDFProcessor, ContextualChunker, ProcessedDocument, DocumentChunk
from embeddings import EmbeddingGenerator, SupabaseVectorStore, EmbeddingPipeline, EmbeddedChunk
from retriever import HybridRetriever, SupabaseRetriever, LocalHybridRetriever, RetrievalResult
from rag_pipeline import RAGSystem, RAGModule, DocumentIngestionPipeline, RAGResponse
from evaluation import (
    RAGASEvaluator,
    DSPyEvaluator,
    LLMJudgeEvaluator,
    PipelineEvaluator,
    EvalSample,
    EvalResult,
    ComponentScores,
)

__all__ = [
    # PDF Processing
    "PDFProcessor",
    "ContextualChunker", 
    "ProcessedDocument",
    "DocumentChunk",
    
    # Embeddings
    "EmbeddingGenerator",
    "SupabaseVectorStore",
    "EmbeddingPipeline",
    "EmbeddedChunk",
    
    # Retrieval
    "HybridRetriever",
    "SupabaseRetriever",
    "LocalHybridRetriever",
    "RetrievalResult",
    
    # RAG Pipeline
    "RAGSystem",
    "RAGModule",
    "DocumentIngestionPipeline",
    "RAGResponse",
    
    # Evaluation
    "DSPyEvaluator",
    "RAGASEvaluator",
    "LLMJudgeEvaluator",
    "PipelineEvaluator",
    "EvalSample",
    "EvalResult",
    "ComponentScores",
]

__version__ = "0.1.0"

