"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Hybrid Retriever: BM25 + pgvector Semantic Search + Reranking               ║
║                                                                              ║
║  Combines keyword-based (BM25) and semantic (vector) search using            ║
║  Reciprocal Rank Fusion (RRF) for optimal retrieval performance.            ║
║  Optional Cohere reranking for improved precision.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from supabase import create_client, Client
from dotenv import load_dotenv

from embeddings import EmbeddingGenerator

# Optional Cohere reranker
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with scores and metadata."""
    content: str
    source: str
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    # Scores from different retrieval methods
    similarity_score: float = 0.0  # Vector similarity (0-1)
    bm25_score: float = 0.0  # BM25 keyword score
    rrf_score: float = 0.0  # Combined RRF score
    
    # Ranking info
    vector_rank: int = 0
    bm25_rank: int = 0
    final_rank: int = 0


class SupabaseRetriever:
    """
    Vector-only retriever using Supabase pgvector.

    Fast and effective for semantic search when you don't need
    keyword matching capabilities.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = 1536,  # Truncate for Supabase HNSW limit
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        table_name: str = "documents",
    ):
        """
        Initialize the Supabase retriever.

        Args:
            model_name: Embedding model for query embedding. Options:
                - "text-embedding-3-small" (OpenAI, recommended)
                - "text-embedding-3-large" (OpenAI, highest quality)
                - "all-MiniLM-L6-v2" (local, free)
            embedding_dimensions: Optional dimension override for OpenAI models.
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
            table_name: Name of the documents table
        """
        load_dotenv()

        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found!\n"
                "Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )

        self.client: Client = create_client(self.url, self.key)
        self.table_name = table_name

        # Initialize embedding model using factory
        kwargs = {}
        if embedding_dimensions and model_name.startswith("text-embedding-"):
            kwargs["dimensions"] = embedding_dimensions

        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = EmbeddingGenerator(model_name=model_name, **kwargs)

        logger.info("SupabaseRetriever initialized")
    
    def __call__(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents similar to the query.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult objects
        """
        return self.retrieve(query, k=k, threshold=threshold)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents similar to the query.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of RetrievalResult objects
        """
        # Embed the query using the configured embedder
        query_embedding = self.embedder.embed(query)

        # Search in Supabase
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": k,
            }
        ).execute()

        # Convert to RetrievalResult objects
        results = []
        for i, doc in enumerate(result.data or []):
            results.append(RetrievalResult(
                content=doc["content"],
                source=doc.get("source", ""),
                section=doc.get("section"),
                metadata=doc.get("metadata", {}),
                similarity_score=doc.get("similarity", 0.0),
                vector_rank=i + 1,
                final_rank=i + 1,
                rrf_score=doc.get("similarity", 0.0),
            ))

        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search with vector similarity.
    
    How Hybrid Retrieval Works:
    ═══════════════════════════
    
    Query: "How do I authenticate with the API?"
    
    BM25 Search:                      Vector Search:
    ┌────────────────────────┐       ┌────────────────────────┐
    │ Finds docs with exact  │       │ Finds semantically     │
    │ terms: "authenticate", │       │ similar docs about     │
    │ "API"                  │       │ authentication         │
    └────────────────────────┘       └────────────────────────┘
            │                                  │
            └──────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Reciprocal     │
              │  Rank Fusion    │
              │  (RRF)          │
              └─────────────────┘
                       │
                       ▼
              Best of both worlds!
    
    Why Hybrid is Better:
    - BM25 excels at exact term matching ("Bearer token", "403 error")
    - Vector search captures semantic meaning ("login" ≈ "authenticate")
    - RRF combines rankings without needing score normalization
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = 1536,  # Truncate for Supabase HNSW limit
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        table_name: str = "documents",
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        use_reranker: bool = True,  # Cohere reranking enabled
        rerank_model: str = "rerank-v3.5",
        over_retrieve_factor: int = 4,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            model_name: Embedding model for embeddings. Options:
                - "text-embedding-3-small" (OpenAI, recommended)
                - "text-embedding-3-large" (OpenAI, highest quality)
                - "all-MiniLM-L6-v2" (local, free)
            embedding_dimensions: Optional dimension override for OpenAI models.
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
            table_name: Name of the documents table
            bm25_weight: Weight for BM25 scores in fusion (0-1)
            vector_weight: Weight for vector scores in fusion (0-1)
            use_reranker: Enable Cohere reranking for improved precision
            rerank_model: Cohere rerank model ("rerank-v3.5" or "rerank-english-v3.0")
            over_retrieve_factor: How many more candidates to retrieve before reranking
        """
        load_dotenv()

        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found!\n"
                "Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )

        self.client: Client = create_client(self.url, self.key)
        self.table_name = table_name
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.over_retrieve_factor = over_retrieve_factor

        # Initialize embedding model using factory
        kwargs = {}
        if embedding_dimensions and model_name.startswith("text-embedding-"):
            kwargs["dimensions"] = embedding_dimensions

        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = EmbeddingGenerator(model_name=model_name, **kwargs)

        # BM25 index - will be built on first query or explicit call
        self._bm25_index = None
        self._documents_cache = None

        # Initialize Cohere reranker if available and requested
        self.use_reranker = use_reranker and COHERE_AVAILABLE
        self.rerank_model = rerank_model
        self._reranker = None

        if use_reranker:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if COHERE_AVAILABLE and cohere_api_key:
                self._reranker = cohere.Client(api_key=cohere_api_key)
                logger.info(f"Cohere reranker enabled: {rerank_model}")
            elif not COHERE_AVAILABLE:
                logger.warning("Cohere not installed. Install with: pip install cohere")
                self.use_reranker = False
            else:
                logger.warning("COHERE_API_KEY not found. Reranking disabled.")
                self.use_reranker = False

        logger.info("HybridRetriever initialized")
    
    def _load_documents(self) -> list[dict]:
        """Load all documents from Supabase for BM25 indexing."""
        if self._documents_cache is not None:
            return self._documents_cache
        
        logger.info("Loading documents from Supabase for BM25 indexing...")
        
        # Fetch all documents (paginated for large collections)
        all_docs = []
        page_size = 1000
        offset = 0
        
        while True:
            result = self.client.table(self.table_name)\
                .select("id, content, source, section, metadata")\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not result.data:
                break
            
            all_docs.extend(result.data)
            
            if len(result.data) < page_size:
                break
            
            offset += page_size
        
        self._documents_cache = all_docs
        logger.info(f"Loaded {len(all_docs)} documents")
        return all_docs
    
    def _build_bm25_index(self):
        """Build BM25 index from documents."""
        if self._bm25_index is not None:
            return
        
        docs = self._load_documents()
        
        # Tokenize documents (simple word splitting)
        tokenized = [doc["content"].lower().split() for doc in docs]
        
        self._bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index built")
    
    def refresh_index(self):
        """Refresh the BM25 index with latest documents from Supabase."""
        self._documents_cache = None
        self._bm25_index = None
        self._build_bm25_index()
    
    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """
        Perform BM25 keyword search.

        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        self._build_bm25_index()

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get indices sorted by score
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        return scored_indices[:k * self.over_retrieve_factor]  # Get more for fusion/reranking
    
    def _vector_search(
        self,
        query: str,
        k: int,
        threshold: float = 0.2,  # Balanced threshold for recall
    ) -> list[tuple[int, float, dict]]:
        """
        Perform vector similarity search.

        Returns:
            List of (doc_index, similarity, doc_data) tuples
        """
        # Embed the query using the configured embedder
        query_embedding = self.embedder.embed(query)

        # Search in Supabase - get more candidates for fusion/reranking
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": k * self.over_retrieve_factor,
            }
        ).execute()

        # Map results to document indices
        docs = self._load_documents()
        doc_id_to_idx = {doc["id"]: i for i, doc in enumerate(docs)}

        results = []
        for doc in result.data or []:
            idx = doc_id_to_idx.get(doc.get("id"))
            if idx is not None:
                results.append((idx, doc.get("similarity", 0.0), doc))

        return results

    def _rerank(
        self,
        query: str,
        results: list["RetrievalResult"],
        top_n: int,
    ) -> list["RetrievalResult"]:
        """
        Rerank results using Cohere reranker.

        Args:
            query: Original search query
            results: List of RetrievalResult to rerank
            top_n: Number of results to return after reranking

        Returns:
            Reranked list of RetrievalResult
        """
        import time

        if not self._reranker or not results:
            return results[:top_n]

        try:
            # Rate limiting for trial tier (10 calls/min = 1 call per 6 seconds)
            if hasattr(self, '_last_rerank_time'):
                elapsed = time.time() - self._last_rerank_time
                if elapsed < 6.5:  # Wait to avoid rate limiting
                    time.sleep(6.5 - elapsed)
            self._last_rerank_time = time.time()

            # Prepare documents for reranking
            documents = [r.content for r in results]

            # Call Cohere rerank API
            rerank_response = self._reranker.rerank(
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                model=self.rerank_model,
            )

            # Reorder results based on reranking
            reranked = []
            for i, item in enumerate(rerank_response.results):
                result = results[item.index]
                result.final_rank = i + 1
                result.rrf_score = item.relevance_score  # Use rerank score
                reranked.append(result)

            logger.info(f"Reranked {len(results)} -> {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return results[:top_n]
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        vector_results: list[tuple[int, float, dict]],
        k_constant: int = 60,
    ) -> list[tuple[int, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        
        RRF Formula: score(d) = Σ 1/(k + rank(d))
        
        Why RRF is great:
        - Doesn't require score normalization
        - Robust to score distribution differences
        - The k constant prevents top-ranked items from dominating
        
        Args:
            bm25_results: List of (doc_idx, score) from BM25
            vector_results: List of (doc_idx, similarity, doc_data) from vector search
            k_constant: RRF constant (default 60 works well in practice)
            
        Returns:
            List of (doc_idx, rrf_score) sorted by score descending
        """
        scores = {}
        
        # Add weighted BM25 contribution
        for rank, (idx, _) in enumerate(bm25_results):
            rrf = 1 / (k_constant + rank + 1)
            scores[idx] = scores.get(idx, 0) + self.bm25_weight * rrf
        
        # Add weighted vector contribution
        for rank, (idx, _, _) in enumerate(vector_results):
            rrf = 1 / (k_constant + rank + 1)
            scores[idx] = scores.get(idx, 0) + self.vector_weight * rrf
        
        # Sort by combined score
        results = [(idx, score) for idx, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def __call__(
        self,
        query: str,
        k: int = 5,
        vector_threshold: float = 0.2,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents using hybrid search with optional reranking.

        Args:
            query: Search query
            k: Number of results to return
            vector_threshold: Minimum similarity for vector search

        Returns:
            List of RetrievalResult objects
        """
        return self.retrieve(query, k=k, vector_threshold=vector_threshold)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        vector_threshold: float = 0.2,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents using hybrid search with optional reranking.

        Args:
            query: Search query
            k: Number of results to return
            vector_threshold: Minimum similarity for vector search

        Returns:
            List of RetrievalResult objects
        """
        # Get results from both methods (over-retrieve for reranking)
        bm25_results = self._bm25_search(query, k)
        vector_results = self._vector_search(query, k, threshold=vector_threshold)

        # Create lookup maps for ranks and scores
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        bm25_scores = {idx: score for idx, score in bm25_results}

        vector_ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(vector_results)}
        vector_data = {idx: (sim, doc) for idx, sim, doc in vector_results}

        # Combine with RRF
        fused_results = self._reciprocal_rank_fusion(bm25_results, vector_results)

        # Build intermediate results (more than k for reranking)
        docs = self._load_documents()
        candidates = []

        # Get more candidates if reranking is enabled
        num_candidates = k * self.over_retrieve_factor if self.use_reranker else k

        for rank, (idx, rrf_score) in enumerate(fused_results[:num_candidates]):
            doc = docs[idx]

            # Get vector similarity if available
            sim, _ = vector_data.get(idx, (0.0, None))

            candidates.append(RetrievalResult(
                content=doc["content"],
                source=doc.get("source", ""),
                section=doc.get("section"),
                metadata=doc.get("metadata", {}),
                similarity_score=sim,
                bm25_score=bm25_scores.get(idx, 0.0),
                rrf_score=rrf_score,
                vector_rank=vector_ranks.get(idx, 0),
                bm25_rank=bm25_ranks.get(idx, 0),
                final_rank=rank + 1,
            ))

        # Apply reranking if enabled
        if self.use_reranker and candidates:
            return self._rerank(query, candidates, top_n=k)

        return candidates[:k]


class LocalHybridRetriever:
    """
    Hybrid retriever that works entirely locally without Supabase.

    Useful for:
    - Testing and development
    - Small document collections
    - Offline scenarios

    Note: For local-only usage, prefer "all-MiniLM-L6-v2" to avoid API costs.
    """

    def __init__(
        self,
        documents: list[dict],
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dimensions: Optional[int] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        """
        Initialize with a list of documents.

        Args:
            documents: List of dicts with 'content', 'source', 'section' keys
            model_name: Embedding model. Use "all-MiniLM-L6-v2" for local-only.
            embedding_dimensions: Optional dimension override for OpenAI models.
            bm25_weight: Weight for BM25 in fusion
            vector_weight: Weight for vector search in fusion
        """
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Initialize embedding model using factory
        kwargs = {}
        if embedding_dimensions and model_name.startswith("text-embedding-"):
            kwargs["dimensions"] = embedding_dimensions

        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = EmbeddingGenerator(model_name=model_name, **kwargs)

        # Build BM25 index
        tokenized = [doc["content"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Generate embeddings for all documents
        logger.info("Generating document embeddings...")
        texts = [doc["content"] for doc in documents]
        self.embeddings = self.embedder.embed_batch(texts, show_progress=True)

        logger.info(f"LocalHybridRetriever initialized with {len(documents)} documents")
    
    def __call__(
        self,
        query: str,
        k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve documents using hybrid search."""
        return self.retrieve(query, k=k)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        import numpy as np

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        # Vector search using the configured embedder
        query_embedding = np.array(self.embedder.embed(query))
        embeddings_array = np.array(self.embeddings)
        similarities = np.dot(embeddings_array, query_embedding)
        vector_ranked = np.argsort(similarities)[::-1]

        # RRF fusion
        k_constant = 60
        scores = {}

        for rank, idx in enumerate(bm25_ranked[:k * 2]):
            scores[idx] = scores.get(idx, 0) + self.bm25_weight / (k_constant + rank + 1)

        for rank, idx in enumerate(vector_ranked[:k * 2]):
            scores[idx] = scores.get(idx, 0) + self.vector_weight / (k_constant + rank + 1)

        # Sort and build results
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        bm25_rank_map = {idx: r + 1 for r, idx in enumerate(bm25_ranked)}
        vector_rank_map = {idx: r + 1 for r, idx in enumerate(vector_ranked)}

        for rank, (idx, rrf_score) in enumerate(fused[:k]):
            doc = self.documents[idx]
            results.append(RetrievalResult(
                content=doc["content"],
                source=doc.get("source", ""),
                section=doc.get("section"),
                metadata=doc.get("metadata", {}),
                similarity_score=float(similarities[idx]),
                bm25_score=float(bm25_scores[idx]),
                rrf_score=rrf_score,
                vector_rank=vector_rank_map.get(idx, 0),
                bm25_rank=bm25_rank_map.get(idx, 0),
                final_rank=rank + 1,
            ))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for testing retrieval."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--count", type=int, default=5, help="Number of results")
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid",
                        help="Retrieval mode")
    
    args = parser.parse_args()
    
    if args.mode == "hybrid":
        retriever = HybridRetriever()
    else:
        retriever = SupabaseRetriever()
    
    results = retriever(args.query, k=args.count)
    
    print(f"\nSearch results for: '{args.query}'")
    print("=" * 70)
    
    for result in results:
        print(f"\n[Rank {result.final_rank}]")
        print(f"  Source: {result.source}")
        print(f"  Section: {result.section or 'N/A'}")
        if args.mode == "hybrid":
            print(f"  Scores: RRF={result.rrf_score:.4f}, "
                  f"Vector={result.similarity_score:.4f}, "
                  f"BM25={result.bm25_score:.2f}")
            print(f"  Ranks: Vector=#{result.vector_rank}, BM25=#{result.bm25_rank}")
        else:
            print(f"  Similarity: {result.similarity_score:.4f}")
        print(f"  Content: {result.content[:200]}...")


if __name__ == "__main__":
    main()

