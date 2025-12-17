"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DSPy RAG Pipeline with DeepSeek / Groq / Gemini                            ║
║                                                                              ║
║  Complete RAG system combining:                                              ║
║  - PDF processing with Docling                                               ║
║  - Contextual chunking                                                       ║
║  - Hybrid retrieval (BM25 + pgvector)                                       ║
║  - Generation with DeepSeek (chat/reasoner), Groq, or Gemini                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

import dspy
from dotenv import load_dotenv

from pdf_processor import PDFProcessor, ContextualChunker, DocumentChunk
from embeddings import EmbeddingPipeline, EmbeddingGenerator
from retriever import HybridRetriever, SupabaseRetriever, RetrievalResult

# Optional faithful RAG module
try:
    from faithful_rag import FaithfulRAGModule, FaithfulRAGModuleFast, FaithfulRAGResponse
    FAITHFUL_RAG_AVAILABLE = True
except ImportError:
    FAITHFUL_RAG_AVAILABLE = False

# Optional MLflow integration
try:
    from mlflow_tracking import MLflowTracker, MLflowConfig
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def setup_groq_lm(
    model: str = "moonshotai/kimi-k2-instruct",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dspy.LM:
    """
    Configure DSPy with Groq's kimi-k2-instruct model.
    
    Groq provides:
    - Extremely fast inference (LPU architecture)
    - Multiple model options
    - OpenAI-compatible API
    
    Models available on Groq:
    - moonshotai/kimi-k2-instruct (powerful reasoning)
    - llama-3.3-70b-versatile
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    
    Args:
        model: Model name to use
        temperature: Generation temperature (0-2)
        max_tokens: Maximum tokens in response
        
    Returns:
        Configured DSPy language model
    """
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "Get your API key at: https://console.groq.com/keys\n"
            "Then set it in your .env file."
        )
    
    # DSPy uses the format: provider/model
    lm = dspy.LM(
        model=f"groq/{model}",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    dspy.configure(lm=lm)
    
    logger.info(f"Configured DSPy with Groq: {model}")
    return lm


def setup_gemini_lm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
) -> dspy.LM:
    """
    Alternative: Configure DSPy with Google Gemini.

    Args:
        model: Gemini model name
        temperature: Generation temperature

    Returns:
        Configured DSPy language model
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found!\n"
            "Get your API key at: https://aistudio.google.com/apikey"
        )

    lm = dspy.LM(
        model=f"gemini/{model}",
        api_key=api_key,
        temperature=temperature,
    )

    dspy.configure(lm=lm)

    logger.info(f"Configured DSPy with Gemini: {model}")
    return lm


def setup_deepseek_lm(
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dspy.LM:
    """
    Configure DSPy with DeepSeek API.

    DeepSeek provides:
    - Pay-as-you-go pricing with no rate limits
    - OpenAI-compatible API
    - Two model variants

    Models available:
    - deepseek-chat: Fast, general-purpose model
    - deepseek-reasoner: Advanced reasoning model (R1)

    Args:
        model: Model name ("deepseek-chat" or "deepseek-reasoner")
        temperature: Generation temperature (0-2)
        max_tokens: Maximum tokens in response

    Returns:
        Configured DSPy language model
    """
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found!\n"
            "Get your API key at: https://platform.deepseek.com/api_keys\n"
            "Then set it in your .env file."
        )

    # DeepSeek uses OpenAI-compatible API
    lm = dspy.LM(
        model=f"openai/{model}",
        api_key=api_key,
        api_base="https://api.deepseek.com",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    dspy.configure(lm=lm)

    logger.info(f"Configured DSPy with DeepSeek: {model}")
    return lm


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy Signatures
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerQuestion(dspy.Signature):
    """Answer a question accurately using the provided context. 
    If the context doesn't contain enough information, say so."""
    
    context = dspy.InputField(
        desc="Retrieved document chunks relevant to the question"
    )
    question = dspy.InputField(
        desc="The user's question"
    )
    
    answer = dspy.OutputField(
        desc="A clear, accurate answer based on the context"
    )


class AnswerWithReasoning(dspy.Signature):
    """Answer a question by first reasoning through the relevant context.
    Cite specific information from the context to support your answer."""
    
    context = dspy.InputField(
        desc="Retrieved document chunks relevant to the question"
    )
    question = dspy.InputField(
        desc="The user's question"
    )
    
    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning through the context"
    )
    answer = dspy.OutputField(
        desc="A clear, accurate answer citing the context"
    )
    sources = dspy.OutputField(
        desc="List of sources used (document names/sections)"
    )


class GenerateDocumentSummary(dspy.Signature):
    """Generate a brief summary of a document for contextual chunking."""
    
    title = dspy.InputField(desc="Document title")
    content_preview = dspy.InputField(desc="First ~1000 characters of the document")
    
    summary = dspy.OutputField(
        desc="2-3 sentence summary of what this document covers"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Module
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    question: str
    answer: str
    reasoning: Optional[str] = None
    sources: Optional[str] = None
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    context: str = ""


class RAGModule(dspy.Module):
    """
    Complete RAG module with retrieval and generation.
    
    Architecture:
    ═════════════
    
    User Question
          │
          ▼
    ┌─────────────────┐
    │ Hybrid Retriever│  ← BM25 + Vector search
    └────────┬────────┘
             │
             ▼
    Retrieved Chunks (with RRF ranking)
             │
             ▼
    ┌─────────────────┐
    │ Context Builder │  ← Format chunks for LLM
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ LLM Generator   │  ← Chain-of-Thought reasoning
    │ (Groq kimi-k2)  │
    └────────┬────────┘
             │
             ▼
    RAGResponse (answer + sources + reasoning)
    """
    
    def __init__(
        self,
        retriever,
        use_reasoning: bool = True,
        k: int = 5,
    ):
        """
        Initialize the RAG module.
        
        Args:
            retriever: Retriever instance (HybridRetriever or SupabaseRetriever)
            use_reasoning: Whether to include step-by-step reasoning
            k: Number of documents to retrieve
        """
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.use_reasoning = use_reasoning
        
        if use_reasoning:
            self.generate = dspy.ChainOfThought(AnswerWithReasoning)
        else:
            self.generate = dspy.Predict(AnswerQuestion)
    
    def _build_context(self, chunks: list[RetrievalResult]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.source
            if chunk.section:
                source_info = f"{chunk.source} - {chunk.section}"
            
            context_parts.append(
                f"[Source {i}: {source_info}]\n"
                f"{chunk.content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def forward(self, question: str) -> RAGResponse:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            RAGResponse with answer and metadata
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retriever(question, k=self.k)
        
        # Step 2: Build context
        context = self._build_context(chunks)
        
        # Step 3: Generate answer
        result = self.generate(context=context, question=question)
        
        # Step 4: Build response
        response = RAGResponse(
            question=question,
            answer=result.answer,
            reasoning=getattr(result, 'reasoning', None),
            sources=getattr(result, 'sources', None),
            retrieved_chunks=chunks,
            context=context,
        )
        
        return response
    
    def __call__(self, question: str) -> RAGResponse:
        """Shorthand for forward()."""
        return self.forward(question)


# ═══════════════════════════════════════════════════════════════════════════════
# Document Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentIngestionPipeline:
    """
    Complete pipeline for ingesting documents into the RAG system.

    Pipeline:
    PDF → Docling → Chunks → Context → Embeddings → Supabase
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_contextual: bool = True,
        chunk_method: str = "semantic",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = 1536,  # Truncate for Supabase HNSW limit
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            enable_ocr: Enable OCR for scanned documents
            enable_contextual: Add LLM-generated context to chunks
            chunk_method: Chunking strategy ("semantic", "fixed", "paragraph")
            embedding_model: Model for embeddings. Options:
                - "text-embedding-3-small" (OpenAI, recommended)
                - "text-embedding-3-large" (OpenAI, highest quality)
                - "all-MiniLM-L6-v2" (local, free)
            embedding_dimensions: Optional dimension override for OpenAI models.
                                 Use 512 or 1024 for cost/storage savings.
        """
        self.pdf_processor = PDFProcessor(enable_ocr=enable_ocr)
        self.chunk_method = chunk_method
        self.enable_contextual = enable_contextual

        if enable_contextual:
            self.contextualizer = ContextualChunker()
            self.summarizer = dspy.Predict(GenerateDocumentSummary)

        self.embedding_pipeline = EmbeddingPipeline(
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )

        logger.info("DocumentIngestionPipeline initialized")
    
    def _generate_summary(self, doc) -> str:
        """Generate a summary for contextual chunking."""
        try:
            result = self.summarizer(
                title=doc.metadata.get('title', doc.source),
                content_preview=doc.markdown[:1000],
            )
            return result.summary
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return f"Document: {doc.metadata.get('title', doc.source)}"
    
    def ingest(
        self,
        source: str | Path,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest a single document.
        
        Args:
            source: Path to PDF or URL
            metadata: Additional metadata to attach
            
        Returns:
            Number of chunks stored
        """
        logger.info(f"Ingesting: {source}")
        
        # Step 1: Process PDF
        doc = self.pdf_processor.process(source)
        
        if metadata:
            doc.metadata.update(metadata)
        
        # Step 2: Chunk the document
        chunks = self.pdf_processor.chunk_document(doc, method=self.chunk_method)
        
        # Step 3: Add contextual prefixes (optional)
        if self.enable_contextual:
            summary = self._generate_summary(doc)
            chunks = self.contextualizer.add_context(chunks, summary)
        
        # Step 4: Embed and store
        count = self.embedding_pipeline.process_and_store(chunks)
        
        logger.info(f"Ingested {count} chunks from {source}")
        return count
    
    def ingest_batch(
        self,
        sources: list[str | Path],
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest multiple documents.
        
        Args:
            sources: List of paths or URLs
            metadata: Metadata to attach to all documents
            
        Returns:
            Total number of chunks stored
        """
        total = 0
        for source in sources:
            try:
                count = self.ingest(source, metadata)
                total += count
            except Exception as e:
                logger.error(f"Failed to ingest {source}: {e}")
                continue
        
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# Complete RAG System
# ═══════════════════════════════════════════════════════════════════════════════

class RAGSystem:
    """
    Complete RAG system combining all components.
    
    Usage:
        # Initialize
        rag = RAGSystem()
        
        # Ingest documents
        rag.ingest("document.pdf")
        rag.ingest_batch(["doc1.pdf", "doc2.pdf"])
        
        # Query
        response = rag.query("What is the main topic?")
        print(response.answer)
    """
    
    def __init__(
        self,
        llm_provider: str = "deepseek",  # "deepseek", "groq", or "gemini"
        llm_model: Optional[str] = None,
        hybrid_retrieval: bool = True,
        enable_ocr: bool = True,
        enable_contextual: bool = True,
        k: int = 5,
        save_questions_to_faq: bool = True,
        faq_source: str = "faq",
        enable_mlflow: bool = False,
        mlflow_config: Optional["MLflowConfig"] = None,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = 1536,  # Truncate for Supabase HNSW limit
        faithful_mode: Optional[str] = None,  # None, "fast", or "full"
    ):
        """
        Initialize the complete RAG system.

        Args:
            llm_provider: LLM provider ("deepseek", "groq", or "gemini")
            llm_model: Specific model name (uses provider default if None)
            hybrid_retrieval: Use hybrid (BM25 + vector) or vector-only
            enable_ocr: Enable OCR for PDF processing
            enable_contextual: Enable contextual chunking
            k: Number of documents to retrieve
            save_questions_to_faq: Persist user Q&A pairs as FAQ entries
            faq_source: Source label stored with FAQ chunks
            enable_mlflow: Enable MLflow tracking for queries and ingestion
            mlflow_config: Optional MLflow configuration
            embedding_model: Model for embeddings. Options:
                - "text-embedding-3-small" (OpenAI, ~$0.02/1M tokens, recommended)
                - "text-embedding-3-large" (OpenAI, ~$0.13/1M tokens, highest quality)
                - "all-MiniLM-L6-v2" (local, free, lower quality)
            embedding_dimensions: Optional dimension override for OpenAI models.
                                 Use 512 or 1024 for cost/storage savings.
            faithful_mode: Enable claim verification for higher faithfulness.
                - None: Standard generation (default)
                - "fast": Batch claim verification (3 LLM calls)
                - "full": Individual claim verification (N+3 LLM calls, most accurate)
        """
        load_dotenv()

        self.llm_provider = llm_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        # Setup LLM
        if llm_provider == "deepseek":
            model = llm_model or "deepseek-chat"
            self.lm = setup_deepseek_lm(model=model)
        elif llm_provider == "groq":
            model = llm_model or "moonshotai/kimi-k2-instruct"
            self.lm = setup_groq_lm(model=model)
        else:
            model = llm_model or "gemini-2.5-flash"
            self.lm = setup_gemini_lm(model=model)

        self.llm_model = model

        # Setup retriever with embedding configuration
        if hybrid_retrieval:
            self.retriever = HybridRetriever(
                model_name=embedding_model,
                embedding_dimensions=embedding_dimensions,
            )
        else:
            self.retriever = SupabaseRetriever(
                model_name=embedding_model,
                embedding_dimensions=embedding_dimensions,
            )

        # Setup RAG module
        self.faithful_mode = faithful_mode
        if faithful_mode and FAITHFUL_RAG_AVAILABLE:
            if faithful_mode == "fast":
                self.rag = FaithfulRAGModuleFast(retriever=self.retriever, k=k)
            else:  # "full"
                self.rag = FaithfulRAGModule(retriever=self.retriever, k=k, mode="verify")
            logger.info(f"Using FaithfulRAG module (mode={faithful_mode})")
        elif faithful_mode and not FAITHFUL_RAG_AVAILABLE:
            logger.warning("faithful_mode requested but faithful_rag.py not found, using standard RAG")
            self.rag = RAGModule(retriever=self.retriever, use_reasoning=True, k=k)
        else:
            self.rag = RAGModule(retriever=self.retriever, use_reasoning=True, k=k)

        # Setup ingestion pipeline
        self.ingestion = DocumentIngestionPipeline(
            enable_ocr=enable_ocr,
            enable_contextual=enable_contextual,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
        self.save_questions_to_faq = save_questions_to_faq
        self.faq_source = faq_source

        # MLflow tracking
        self._mlflow_tracker = None
        if enable_mlflow:
            if not MLFLOW_AVAILABLE:
                logger.warning("MLflow not available. Install with: pip install mlflow")
            else:
                self._mlflow_tracker = MLflowTracker(mlflow_config)
                # Log initial model configuration
                self._mlflow_tracker.log_model_config({
                    "llm_provider": llm_provider,
                    "llm_model": model,
                    "hybrid_retrieval": hybrid_retrieval,
                    "k": k,
                    "save_questions_to_faq": save_questions_to_faq,
                    "embedding_model": embedding_model,
                    "embedding_dimensions": embedding_dimensions,
                })
                logger.info(f"MLflow tracking enabled: {self._mlflow_tracker.get_experiment_url()}")

        logger.info(f"RAGSystem initialized with {llm_provider}/{model}, embeddings: {embedding_model}")
    
    def _store_faq_interaction(self, response: RAGResponse):
        """Persist the latest Q&A pair as a FAQ-style chunk for future retrieval."""
        if not self.save_questions_to_faq:
            return
        if not response.question or not response.answer:
            return
        
        content = (
            f"Question: {response.question.strip()}\n"
            f"Answer: {response.answer.strip()}"
        )
        if response.sources:
            content = f"{content}\nSources: {response.sources}"
        
        metadata = {
            "type": "faq",
            "question": response.question,
            "answer_preview": response.answer[:240],
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        if response.sources:
            metadata["sources"] = response.sources
        
        try:
            chunk = DocumentChunk(
                content=content,
                source=self.faq_source,
                section="FAQ",
                metadata=metadata,
            )
            self.ingestion.embedding_pipeline.process_and_store([chunk], batch_size=1)
            
            # Ensure hybrid BM25 cache sees the new FAQ entry
            if hasattr(self.retriever, "refresh_index"):
                self.retriever.refresh_index()
        except Exception as e:
            logger.warning(f"Failed to store FAQ interaction: {e}")
    
    def ingest(self, source: str | Path, **kwargs) -> int:
        """Ingest a single document."""
        if self._mlflow_tracker:
            with self._mlflow_tracker.track_ingestion(str(source)):
                count = self.ingestion.ingest(source, **kwargs)
                self._mlflow_tracker.log_ingestion_result(count, str(source), kwargs.get('metadata'))
                return count
        return self.ingestion.ingest(source, **kwargs)

    def ingest_batch(self, sources: list[str | Path], **kwargs) -> int:
        """Ingest multiple documents."""
        return self.ingestion.ingest_batch(sources, **kwargs)
    
    def query(self, question: str) -> RAGResponse:
        """Query the RAG system."""
        start_time = time.time()

        response = self.rag(question)
        self._store_faq_interaction(response)

        # Track with MLflow if enabled
        if self._mlflow_tracker:
            latency_ms = (time.time() - start_time) * 1000
            with self._mlflow_tracker.track_query(question):
                self._mlflow_tracker.log_query_result(response)
                self._mlflow_tracker.log_metric("latency_ms", latency_ms)

        return response
    
    def refresh_index(self):
        """Refresh the BM25 index after adding new documents."""
        if hasattr(self.retriever, 'refresh_index'):
            self.retriever.refresh_index()
    
    def interactive(self):
        """Start an interactive query session."""
        print("\n" + "=" * 60)
        print("RAG System Interactive Mode")
        print("Type 'quit' to exit, 'refresh' to rebuild index")
        print("=" * 60 + "\n")
        
        while True:
            try:
                question = input("❓ Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not question:
                continue
            if question.lower() in ('quit', 'exit', 'q'):
                break
            if question.lower() == 'refresh':
                self.refresh_index()
                print("Index refreshed!\n")
                continue
            
            response = self.query(question)

            print(f"\n💬 Answer: {response.answer}")

            if response.reasoning:
                print(f"\n🔍 Reasoning: {response.reasoning}")

            if response.sources:
                print(f"\n📚 Sources: {response.sources}")

            # Show faithfulness info if using faithful mode
            if hasattr(response, 'faithfulness_score'):
                print(f"\n✅ Faithfulness: {response.faithfulness_score:.1%}")
                if response.claims_unsupported:
                    print(f"   ⚠️  {len(response.claims_unsupported)} unsupported claims removed")

            print(f"\n📄 Retrieved {len(response.retrieved_chunks)} chunks:")
            for chunk in response.retrieved_chunks[:3]:
                print(f"   - {chunk.source}")
                if hasattr(chunk, 'section') and chunk.section:
                    print(f"     Section: {chunk.section}")

            print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for the RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("sources", nargs="+", help="PDF files or URLs to ingest")
    ingest_parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    ingest_parser.add_argument("--no-contextual", action="store_true",
                               help="Disable contextual chunking")
    ingest_parser.add_argument("--mlflow", action="store_true",
                               help="Enable MLflow tracking")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("-k", "--count", type=int, default=5,
                              help="Number of documents to retrieve")
    query_parser.add_argument("--no-save-faq", action="store_true",
                              help="Do not store this Q&A as a FAQ chunk")
    query_parser.add_argument("--mlflow", action="store_true",
                              help="Enable MLflow tracking")
    query_parser.add_argument("--faithful", choices=["fast", "full"],
                              help="Enable claim verification (fast=batch, full=individual)")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    interactive_parser.add_argument("--no-save-faq", action="store_true",
                                    help="Do not store Q&A from this session")
    interactive_parser.add_argument("--mlflow", action="store_true",
                                    help="Enable MLflow tracking")
    interactive_parser.add_argument("--faithful", choices=["fast", "full"],
                                    help="Enable claim verification (fast=batch, full=individual)")
    
    # Parse args
    args = parser.parse_args()
    
    if args.command == "ingest":
        rag = RAGSystem(
            enable_ocr=not args.no_ocr,
            enable_contextual=not args.no_contextual,
            enable_mlflow=args.mlflow,
        )

        total = rag.ingest_batch(args.sources)
        print(f"\n✅ Ingested {total} chunks from {len(args.sources)} documents")

    elif args.command == "query":
        rag = RAGSystem(
            k=args.count,
            save_questions_to_faq=not args.no_save_faq,
            enable_mlflow=args.mlflow,
            faithful_mode=getattr(args, 'faithful', None),
        )

        response = rag.query(args.question)

        print(f"\n❓ Question: {args.question}")
        print(f"\n💬 Answer: {response.answer}")

        if response.reasoning:
            print(f"\n🔍 Reasoning: {response.reasoning}")

        if response.sources:
            print(f"\n📚 Sources: {response.sources}")

        # Show faithfulness info if using faithful mode
        if hasattr(response, 'faithfulness_score'):
            print(f"\n✅ Faithfulness: {response.faithfulness_score:.1%}")
            if response.claims_unsupported:
                print(f"   Removed {len(response.claims_unsupported)} unsupported claims:")
                for v in response.claims_unsupported[:3]:
                    print(f"   - {v.claim[:80]}...")

    elif args.command == "interactive":
        rag = RAGSystem(
            save_questions_to_faq=not args.no_save_faq,
            enable_mlflow=args.mlflow,
            faithful_mode=getattr(args, 'faithful', None),
        )
        rag.interactive()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

