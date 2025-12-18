# Production Medical RAG Pipeline

**[Live Demo](https://dspy-supabase-rag.streamlit.app/)** | A reference implementation achieving **98.3% faithfulness** on medical documents. This README explains the concepts, trade-offs, and architectural decisions for building production RAG systems.

```
Documents → Semantic Chunks → Vector Embeddings → Hybrid Retrieval → Reranking → LLM Generation
```

---

## Results

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Faithfulness** | **98.3%** | Only 1.7% of claims aren't grounded in source docs |
| **Answer Relevancy** | **96.0%** | Answers directly address the question asked |
| **Context Precision** | 93.6% | Retrieved chunks are actually relevant |
| **Context Recall** | 93.3% | Found most of the relevant information |

*Tested on 6 medical AI papers: CheXNet, ClinicalBERT, AdvProp, medical imaging, skin cancer classification, COVID forecasting.*

---

## Architecture

### Ingestion Pipeline

```
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐      ┌─────────────┐
│   PDF/DOCX  │ ───▶ │   Docling   │ ───▶ │   Contextual    │ ───▶ │  Supabase   │
│  Documents  │      │   Parser    │      │    Chunking     │      │  pgvector   │
└─────────────┘      └─────────────┘      └─────────────────┘      └─────────────┘
                            │                     │                       │
                     Layout analysis        Add doc context         OpenAI embeddings
                     Table extraction       to each chunk           1536 dimensions
                     OCR for scans          (Anthropic method)      HNSW index
```

### Query Pipeline

```
                              ┌─────────────────────────────────────┐
                              │              User Query             │
                              └──────────────────┬──────────────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
                  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
                  │    BM25     │         │   Vector    │         │   Query     │
                  │  (keyword)  │         │ (semantic)  │         │  Analysis   │
                  └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
                         │                       │                       │
                         └───────────┬───────────┘                       │
                                     ▼                                   │
                         ┌─────────────────────┐                         │
                         │   Reciprocal Rank   │                         │
                         │   Fusion (k=60)     │                         │
                         └──────────┬──────────┘                         │
                                    │                                    │
                                    ▼                                    │
                         ┌─────────────────────┐                         │
                         │   LLM Reranker      │  20 candidates ──▶ 5    │
                         │    (DeepSeek)       │                         │
                         └──────────┬──────────┘                         │
                                    │                                    │
                                    ▼                                    ▼
                         ┌─────────────────────────────────────────────────┐
                         │              Confident RAG Module               │
                         │  ┌─────────────────────────────────────────┐   │
                         │  │  Generate Answer ──▶ Extract Claims ──▶ │   │
                         │  │  Verify Claims ──▶ Compute Confidence   │   │
                         │  └─────────────────────────────────────────┘   │
                         └──────────────────────┬──────────────────────────┘
                                                │
                              ┌─────────────────┴─────────────────┐
                              ▼                                   ▼
                    ┌─────────────────┐                 ┌─────────────────┐
                    │  Confidence ≥   │                 │  Confidence <   │
                    │   Threshold     │                 │   Threshold     │
                    │                 │                 │                 │
                    │  Return Answer  │                 │    Abstain      │
                    │  + Sources      │                 │  "I don't know" │
                    └─────────────────┘                 └─────────────────┘
```

### Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| **PDF Parsing** | Docling | Layout-aware, handles tables, OCR |
| **Embeddings** | OpenAI text-embedding-3-large | Best quality, truncated to 1536d |
| **Vector Store** | Supabase pgvector | Free tier, SQL queries, HNSW |
| **Keyword Search** | BM25 (rank-bm25) | Exact term matching |
| **Reranking** | DeepSeek LLM | No rate limits, cheap |
| **Generation** | DeepSeek chat | No rate limits, good quality |
| **Framework** | DSPy | Structured LLM modules |
| **Evaluation** | RAGAS + GPT-4o-mini | Industry standard, external judge |

### Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Test Set (JSON)                                │
│                    question + expected_answer pairs                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline                                      │
│                     (DeepSeek generation)                                   │
│                                                                             │
│   Question ──▶ Retrieve Context ──▶ Generate Answer ──▶ (answer, contexts) │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAGAS Evaluation                                     │
│                   (GPT-4o-mini as judge)                                    │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   Faithfulness   │  │ Answer Relevancy │  │ Context Precision│          │
│  │                  │  │                  │  │                  │          │
│  │ Extract claims   │  │ Generate reverse │  │ LLM judges each  │          │
│  │ from answer,     │  │ questions from   │  │ retrieved chunk  │          │
│  │ verify each      │  │ answer, compare  │  │ for relevance    │          │
│  │ against context  │  │ to original      │  │                  │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           ▼                     ▼                     ▼                     │
│        98.3%                 96.0%                 93.6%                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   eval_results/*.json   │
                    │   eval_results/*.md     │
                    └─────────────────────────┘
```

**Why separate models?** Using DeepSeek for generation and GPT-4o-mini for evaluation avoids self-evaluation bias (models rate their own outputs higher).

---

## Core Concepts

### What is RAG?

**Problem:** LLMs are trained on public data. They don't know your documents, and they hallucinate when asked about things they don't know.

**Solution:** RAG (Retrieval-Augmented Generation) = find relevant chunks from your docs first, then generate an answer using only that context. The LLM becomes a "reader" rather than relying on memorized knowledge.

**Key insight:** RAG separates *knowledge* (your documents) from *reasoning* (the LLM). You can update knowledge without retraining.

### Why RAG Beats Fine-Tuning

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-tuning** | Knowledge baked in, fast inference | Expensive, can't update easily, still hallucinates |
| **RAG** | Easy to update, citations, auditable | Retrieval can fail, more complex |
| **RAG + Fine-tuning** | Best of both | Most complex, highest cost |

**For most enterprise use cases, RAG alone is sufficient.** Fine-tuning is only worth it when you need the model to learn a new *style* or *behavior*, not just new *facts*.

---

## The Retrieval Problem

The hardest part of RAG is retrieval. If you don't find the right chunks, the LLM can't generate a good answer.

### Vector Search (Semantic)

Converts text to numbers (embeddings) that capture meaning. "Revenue" and "income" have similar vectors.

**How it works:**
1. Each chunk → 1536-dimensional vector (using OpenAI's embedding model)
2. Query → same vector space
3. Find chunks with highest cosine similarity

**Strengths:** Finds semantically similar content even with different words.

**Weaknesses:** Misses exact terms. "OAuth 2.0" might not match "OAuth2" well.

### BM25 Search (Keyword)

Classic information retrieval. Scores documents by term frequency and inverse document frequency.

**Strengths:** Exact matches. Drug names, legal terms, technical jargon.

**Weaknesses:** Misses synonyms. "Heart attack" won't match "myocardial infarction".

### Hybrid Search (What We Use)

Combine both and merge results using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(doc) = 1/(k + rank_bm25) + 1/(k + rank_vector)
```

Where k=60 is a smoothing constant. Documents ranked highly by both methods score highest.

**Why RRF over simple score addition?** Scores from different systems aren't comparable. BM25 might give scores 0-20, vectors give 0-1. RRF uses ranks, which are always comparable.

### Reranking (The Secret Weapon)

Initial retrieval is fast but rough. We retrieve 20 candidates, then use an LLM to rerank them:

```
Retrieve 20 (recall) → Rerank to top 5 (precision)
```

**Why it works:** The reranker sees the full query + each candidate and judges relevance more accurately than embedding similarity alone.

**Cost:** ~10% precision improvement for ~3x latency. Worth it for high-stakes domains.

| Reranker | Cost | Quality | Notes |
|----------|------|---------|-------|
| LLM (DeepSeek) | $0.001/query | Best | No rate limits |
| Cohere | Free tier | Good | 10 req/min limit |
| Cross-encoder | Free | Good | Slower, local |
| None | Free | Baseline | Just RRF |

---

## The Faithfulness Problem

Even with perfect retrieval, LLMs can hallucinate. They might:
- Add details not in the context
- Misinterpret numbers or dates
- Confidently state things they inferred incorrectly

### Measuring Faithfulness

**Faithfulness** = % of claims in the answer that are supported by retrieved context.

How RAGAS computes it:
1. Extract atomic claims from the answer
2. For each claim, check if it's entailed by the context
3. Faithfulness = supported_claims / total_claims

**Target for medical/legal:** >97%. Below that, you're serving hallucinations.

### Strategy 1: Claim Verification (Faithful Mode)

Post-process every answer:
```
Generate Answer → Extract Claims → Verify Each Claim → Rewrite with Only Supported Claims
```

**Trade-off:** +3-4% faithfulness, ~3x latency, may remove relevant-but-unverifiable content.

### Strategy 2: Confidence-Gated RAG (Abstention)

Compute a confidence score and refuse to answer when uncertain:

```python
confidence = 0.5 * faithfulness_score + 0.5 * context_coverage
if confidence < threshold:
    return "I don't have enough information to answer reliably."
```

**Trade-off:** Higher faithfulness on answered questions, but some questions go unanswered.

| Threshold | Faithfulness | Abstention Rate |
|-----------|--------------|-----------------|
| 0.50 | ~95% | ~5% |
| 0.60 | ~98% | ~10% |
| 0.75 | ~99% | ~25% |

**Key insight:** For medical/legal, it's better to say "I don't know" than to hallucinate. Users trust systems that admit uncertainty.

---

## Chunking Strategies

How you split documents matters enormously.

### Naive Chunking

Split every N characters/tokens. Simple but loses context.

**Problem:** A chunk might say "The drug increased survival by 40%" but you don't know which drug.

### Semantic Chunking

Split at natural boundaries (paragraphs, sections). Keeps semantic units together.

### Contextual Chunking (What We Use)

Add document context to each chunk:

```
Before: "The drug increased survival by 40%"
After:  "[From: Phase 3 Trial Results, Drug XYZ vs Placebo]
         The drug increased survival by 40%"
```

Based on [Anthropic's contextual retrieval research](https://www.anthropic.com/news/contextual-retrieval). Improves retrieval significantly because chunks are self-contained.

**Implementation:** For each chunk, ask an LLM to summarize the document context, then prepend it.

---

## Embedding Model Selection

| Model | Dimensions | Cost | Quality | Notes |
|-------|------------|------|---------|-------|
| text-embedding-3-large | 3072 | $0.13/1M | Best | Truncate to 1536 for pgvector |
| text-embedding-3-small | 1536 | $0.02/1M | Good | 6x cheaper |
| all-MiniLM-L6-v2 | 384 | Free | Basic | Local, fast |

**Practical note:** Supabase HNSW index limit is 2000 dimensions. We truncate OpenAI's 3072 → 1536.

**When to use local models:** When you can't send data to external APIs (compliance), or at massive scale where API costs dominate.

---

## Evaluation

### Why Evaluation is Hard

You can't just check if the answer is "correct" because:
1. Many valid ways to answer a question
2. Ground truth is expensive to create
3. RAG answers include reasoning, citations, caveats

### RAGAS Framework

Industry standard for RAG evaluation. Uses LLM-as-judge:

| Metric | What It Measures | How |
|--------|------------------|-----|
| **Answer Relevancy** | Does the answer address the question? | Generate questions from answer, compare to original |
| **Faithfulness** | Is the answer grounded in context? | Extract claims, verify each against context |
| **Context Precision** | Are retrieved chunks relevant? | LLM judges each chunk |
| **Context Recall** | Did we find all relevant info? | Compare to ground truth |

### Avoiding Self-Evaluation Bias

**Critical:** Don't use the same model for generation and evaluation. The model will rate its own outputs higher.

```
Generator: DeepSeek (cheap, no rate limits)
Evaluator: GPT-4o-mini (different model family)
```

### Variance Warning

LLM-as-judge has ~10% variance between runs. Use 50+ samples for stable metrics. Our 98.3% faithfulness means "somewhere between 95-100%" realistically.

---

## Architecture Decisions

### Why DeepSeek?

| Factor | DeepSeek | OpenAI | Anthropic |
|--------|----------|--------|-----------|
| Rate limits | None | Yes | Yes |
| Cost | $0.28/1M in | $3/1M in | $3/1M in |
| Quality | Good | Best | Best |
| Speed | Fast | Fast | Fast |

For RAG, you make many LLM calls (reranking, generation, verification). Rate limits kill throughput. DeepSeek's unlimited API is ideal for development and batch processing.

### Why Supabase pgvector?

| Option | Pros | Cons |
|--------|------|------|
| **Supabase pgvector** | Free tier, SQL, easy | 2000 dim limit |
| Pinecone | Managed, scalable | Expensive at scale |
| Weaviate | Feature-rich | Complex |
| Chroma | Local, simple | Not production-ready |
| FAISS | Fast, local | No persistence |

Supabase gives you a real database with vectors. You can JOIN, filter, and query with SQL. Good enough for <1M vectors.

### Why DSPy?

DSPy structures LLM interactions as composable modules with typed inputs/outputs:

```python
class AnswerQuestion(dspy.Signature):
    context = dspy.InputField(desc="Retrieved documents")
    question = dspy.InputField()
    answer = dspy.OutputField()
```

**Benefits:**
- Automatic prompt optimization
- Testable modules
- Clean separation of concerns

**Alternative:** LangChain. More features, more complexity, harder to debug.

---

## Common Pitfalls

### 1. Retrieval Failures

**Symptom:** Good chunks exist but aren't retrieved.

**Causes:**
- Embedding model mismatch (query vs docs)
- Chunk too large (dilutes signal)
- Missing hybrid search (exact terms missed)

**Fix:** Always use hybrid search. Tune chunk size (512-1024 tokens sweet spot).

### 2. Context Window Overflow

**Symptom:** Truncated context, incomplete answers.

**Cause:** Retrieved too many chunks, exceeded LLM context.

**Fix:** Rerank aggressively. Better to have 5 highly relevant chunks than 20 mediocre ones.

### 3. Hallucination Despite Good Retrieval

**Symptom:** Context contains the answer, but LLM adds fabricated details.

**Cause:** LLM training to be "helpful" overrides faithfulness.

**Fix:** Explicit prompting ("Only use information from the provided context"), claim verification, or confidence gating.

### 4. Evaluation Overfitting

**Symptom:** Great scores on test set, poor real-world performance.

**Cause:** Test questions too similar, tuned hyperparameters to test set.

**Fix:** Held-out test set, adversarial questions, human evaluation sample.

---

## Production Checklist

For a $2B healthcare company, what would they add?

| Component | This Repo | Enterprise |
|-----------|-----------|------------|
| Retrieval | Hybrid + rerank | Same |
| Faithfulness | Claim verification | Same + human review |
| Evaluation | RAGAS automated | + human eval, A/B tests |
| Monitoring | None | Confidence logging, drift detection |
| Compliance | None | HIPAA, audit logs, PII detection |
| Scale | Single instance | Kubernetes, caching, load balancing |
| Feedback | None | User corrections → fine-tuning |

**The architecture is the same.** Enterprise adds human oversight, compliance, and scale.

---

## Key Takeaways for Consulting

1. **Hybrid search is non-negotiable.** Vector-only misses exact terms. BM25-only misses semantics.

2. **Reranking is worth the cost** for high-stakes domains. ~10% precision gain.

3. **Faithfulness > Relevancy** for medical/legal. Better to refuse than hallucinate.

4. **Abstention is a feature.** Confidence thresholds let users trust answered questions.

5. **Evaluation is hard.** Use external judge, large test sets, human spot-checks.

6. **Chunking matters more than model choice.** Contextual chunks with metadata beat naive splits.

7. **The hard part is retrieval.** If you find the right chunks, generation is easy.

8. **Start simple, measure, iterate.** Don't over-engineer. Get baseline metrics first.

---

## Quick Commands

```bash
# Ingest documents
uv run rag_pipeline.py ingest sample_pdfs/*.pdf

# Query (with confidence gating)
uv run rag_pipeline.py query "What is CheXNet?" --faithful confident-lite

# Evaluate
uv run evaluation.py full -f medical_test_set.json --faithful confident-lite

# Interactive mode
uv run rag_pipeline.py interactive
```

---

## File Structure

```
├── app.py                # Streamlit UI (live demo)
├── rag_pipeline.py       # Main RAGSystem orchestrator
├── confident_rag.py      # Confidence-gated RAG with abstention
├── faithful_rag.py       # Claim verification module
├── retriever.py          # Hybrid search + reranking
├── embeddings.py         # Vector embedding generation
├── pdf_processor.py      # Docling PDF parsing
├── evaluation.py         # RAGAS evaluation
├── medical_test_set.json # 15 medical AI questions
├── .streamlit/           # Streamlit config + secrets template
└── eval_results/         # Evaluation outputs
```

---

## License

MIT
