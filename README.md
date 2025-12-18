# DSPy Supabase RAG Pipeline

A production-ready Retrieval-Augmented Generation system with hybrid search, neural reranking, and comprehensive evaluation. Tested on medical AI research papers.

```
Documents → Semantic Chunks → Vector Embeddings → Hybrid Retrieval → Reranking → LLM Generation
```

---

## Performance (Medical Papers)

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **98.3%** | Answers grounded in context (no hallucination) |
| **Answer Relevancy** | **96.0%** | Generated answers address the question |
| **Context Precision** | 93.6% | Retrieved chunks are relevant |
| **Context Recall** | 93.3% | Coverage of relevant information |
| **Overall** | **95.3%** | Weighted average |

*Evaluated on 15 medical AI questions using [RAGAS](https://docs.ragas.io/) with GPT-4o-mini as judge, DeepSeek-chat as generator average with 20 samples.*

*Test corpus: 6 arXiv papers on CheXNet (chest X-ray diagnosis), ClinicalBERT, AdvProp (drug discovery), medical imaging AI, skin cancer classification, and COVID-19 forecasting.*

### High-Stakes Mode (Confident-Lite)

For medical/legal domains, the pipeline includes confidence-gated answering with optional abstention:

```python
rag = RAGSystem(faithful_mode="confident-lite")
```

| Mode | Faithfulness | Relevancy | Use Case |
|------|--------------|-----------|----------|
| Standard | 97.1% | 58.5% | General use |
| **Confident-Lite** | **98.3%** | **96.0%** | Medical/Legal (recommended) |

---

## Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **PDF Parsing** | [Docling](https://github.com/DS4SD/docling) | Layout analysis, table extraction, OCR |
| **Embeddings** | OpenAI text-embedding-3-large | 3072→1536 dims (Supabase HNSW limit) |
| **Vector Store** | Supabase pgvector | PostgreSQL + HNSW index |
| **Keyword Search** | rank-bm25 | BM25Okapi algorithm |
| **Reranking** | DeepSeek LLM | LLM-based reranking (no rate limits) |
| **LLM** | DeepSeek V3.2 (chat/reasoner) | No rate limits, pay-as-you-go |
| **Framework** | DSPy | Structured LLM programming |
| **Evaluation** | RAGAS | Industry-standard metrics |

---

## Architecture

### Ingestion Pipeline

```
PDF/DOCX ──▶ Docling ──▶ Semantic Chunking ──▶ OpenAI Embeddings ──▶ Supabase pgvector
             │              │                    │
             │              │                    └─ text-embedding-3-large
             │              │                       truncated to 1536 dims
             │              │
             │              └─ Contextual chunks with document metadata
             │                 (based on Anthropic's Contextual Retrieval)
             │
             └─ Layout analysis, table detection, OCR for scans
```

### Retrieval Pipeline

```
                           Query
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
         ┌─────────┐                  ┌─────────────┐
         │  BM25   │                  │   Vector    │
         │(keyword)│                  │ (semantic)  │
         └────┬────┘                  └──────┬──────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Reciprocal Rank    │
                  │  Fusion (k=60)      │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  DeepSeek Reranker  │  ← LLM reranking
                  │  (no rate limits)   │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │    DeepSeek LLM     │
                  │ (chat or reasoner)  │
                  └──────────┬──────────┘
                             ▼
                      Answer + Sources
```

### Why Hybrid Search?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| BM25 only | Exact matches ("OAuth 2.0") | Misses synonyms |
| Vector only | Semantic similarity | Misses exact terms |
| **Hybrid** | Both | Minimal gaps |

RRF combines rankings without score normalization:
```
RRF_score(d) = Σ 1/(k + rank(d))  where k=60
```

### Why Reranking?

| Stage | Method | Purpose |
|-------|--------|---------|
| Retrieve | BM25 + Vector | Get 20 candidates (recall) |
| Rerank | DeepSeek LLM | Keep top 5 (precision) |

**Reranker options:**
- `llm` (default): DeepSeek LLM reranking, no rate limits, cheap
- `cohere`: Cohere rerank-v3.5, rate limited on free tier
- `local`: Local cross-encoder, free but slower
- `none`: No reranking, just RRF fusion

---

## Evaluation

### RAGAS Metrics

| Metric | What It Measures | Computation |
|--------|------------------|-------------|
| **Answer Relevancy** | Question addressing | Reverse question similarity |
| **Faithfulness** | Grounding in context | Claim verification |
| **Context Precision** | Retrieved chunk relevance | LLM judges each chunk |
| **Context Recall** | Coverage of ground truth | Claims comparison |

**Answer Relevancy is generally the most important metric** for RAG systems—it measures whether users actually get useful answers to their questions. A system with poor answer relevancy is effectively useless regardless of other scores.

For high-stakes domains (medical, legal, financial), **Faithfulness** becomes equally critical since hallucinated information can cause real harm.

### Score Interpretation

| Range | Quality | Action |
|-------|---------|--------|
| > 80% | Excellent | Production ready |
| 60-80% | Good | Minor tuning |
| 40-60% | Fair | Review retrieval |
| < 40% | Poor | Debug pipeline |

---

## How It Works (Plain English)

### The Problem

Large language models like ChatGPT are trained on public internet data, but they don't know anything about *your* documents, your company's internal reports, or that PDF sitting on your desktop. If you ask them about your specific content, they'll either admit they don't know or make something up.

### The Solution: Retrieval-Augmented Generation (RAG)

RAG is a two-step approach: first *find* the relevant information from your documents, then *generate* an answer using that information. Think of it like giving someone a textbook before an exam, they read the relevant pages, then answer your question based on what they just read.

### The Pipeline, Step by Step

**1. Document Ingestion (Getting your documents ready)**

When you feed a PDF into the system, several things happen:

- **Docling** (a document parser) reads the PDF and understands its structure, tables, headers, paragraphs, even scanned images through OCR. It's smarter than just copying text because it preserves document structure.

- **Chunking** breaks the document into smaller pieces (chunks). You can't feed an entire 200-page PDF to an AI at once, so we split it into digestible paragraphs. Each chunk also gets labeled with where it came from ("From ACME Q3 Report, page 12, Financial Results section").

- **Embeddings** convert each chunk into a list of numbers (a vector). This is the magic that lets us search by meaning rather than just keywords. The word "revenue" and "income" have similar vectors, so a search for one finds the other. We use OpenAI's embedding model for this.

- **Supabase** stores these vectors in a database with a special index (pgvector) that makes searching through thousands of vectors fast.

**2. Retrieval (Finding the right information)**

When you ask a question, the system needs to find which chunks are most relevant. We use two search methods and combine them:

- **Vector search** finds chunks with similar meaning. Ask about "company earnings" and it finds chunks about "revenue" and "profit" even if those exact words weren't in your question.

- **BM25 search** finds exact keyword matches. If you search for "OAuth 2.0", vector search might miss it, but keyword search catches it exactly.

- **Hybrid search** combines both using a technique called Reciprocal Rank Fusion (RRF). If both methods agree a chunk is relevant, it ranks higher. This catches both semantic matches and exact terms.

- **Reranking** is a second pass. The initial search casts a wide net (20 candidates), then DeepSeek LLM looks at each candidate more carefully and picks the best 5. This improves precision by about 10%.

**3. Generation (Creating the answer)**

The final step takes your question plus the top retrieved chunks and sends them to a large language model (we use DeepSeek for reliable inference without rate limits). The model reads the context and writes an answer, citing which chunks it used. **DSPy** is a framework that structures this interaction, it defines exactly what inputs and outputs the model should produce and can optimize prompts automatically.

### Why This Architecture?

Each piece solves a specific problem:

| Component | Problem It Solves |
|-----------|-------------------|
| Docling | PDFs are messy, tables break, scans need OCR |
| Chunking | LLMs have limited input size |
| Embeddings | Need to search by meaning, not just keywords |
| Hybrid search | Keywords miss synonyms, vectors miss exact terms |
| Reranking | Initial search is fast but rough, reranking is slow but precise |
| DSPy | Structured, testable LLM interactions |

### Measuring Quality

How do we know if the system is working well? We use four metrics:

- **Faithfulness (98.3%)**: Is the answer grounded in the retrieved context, or is the AI making things up? Critical for medical/legal trust.

- **Answer Relevancy (96.0%)**: Does the answer actually address what was asked?

- **Context Precision (93.6%)**: Are the retrieved chunks actually relevant, or is the system pulling in junk?

- **Context Recall (93.3%)**: Did we find all the relevant information, or did we miss some?

These are measured automatically using another LLM as a judge (RAGAS framework).

**Important:** Using the same model for generation and evaluation causes self-evaluation bias. We use:
- **Generator:** DeepSeek (cheap, no rate limits)
- **Evaluator:** GPT-4o-mini (different model, accurate judging)

---

### Running Evaluation

**Default model: `deepseek-chat`** (used for both generation and reranking)

```bash
# Normal mode with deepseek-chat (recommended)
uv run evaluation.py full -f test_set.json -o results.json

# With faithful mode (claim verification for high-stakes domains)
uv run evaluation.py full -f test_set.json -o results_faithful.json --faithful fast

# With deepseek-reasoner (5-10x slower, lower faithfulness, not recommended)
uv run evaluation.py full -f test_set.json -o results.json --model deepseek-reasoner
```

**Note:** RAGAS uses LLM-as-judge with ~10% variance. Use 50+ samples for stable metrics.

See [eval.md](eval.md) for detailed results comparing chat vs reasoner and normal vs faithful modes.

---

## Key Features

### Contextual Chunking

Based on [Anthropic's research](https://www.anthropic.com/news/contextual-retrieval) - adds document context to each chunk:

```
Before: "Revenue grew by 15%..."
After:  "[From ACME Q3 2024 Report, Financial Results section]
         Revenue grew by 15%..."
```

### FAQ Capture

Answered Q&A pairs are stored back as retrievable chunks:

```python
rag = RAGSystem(save_questions_to_faq=False)  # Disable if unwanted
```

### Faithful RAG Mode (Claim Verification)

For high-stakes domains (medical, legal, financial) where hallucination is unacceptable, enable claim verification:

```python
rag = RAGSystem(faithful_mode="fast")  # or "full" for individual verification
```

**How it works:**
1. Generate a full, relevant answer (preserves answer quality)
2. Extract atomic claims from the answer
3. Verify each claim against retrieved context
4. Rewrite answer with only supported claims

```bash
# Evaluation with faithful mode
uv run evaluation.py full -f test_set.json -o results.json --faithful fast
```

| Mode | LLM Calls | Use Case |
|------|-----------|----------|
| `fast` | 3 | Batch verification, good balance |
| `full` | N+3 | Individual verification, highest accuracy |

---

## Tuning for High-Stakes Domains

For medical, legal, or financial RAGs where hallucination is unacceptable, you need **faithfulness >97%** while maintaining reasonable answer relevancy. This section covers tuning strategies.

### The Faithfulness vs Relevancy Tradeoff

Standard RAG optimizes for both metrics equally. High-stakes domains must **prioritize faithfulness** even at some cost to relevancy:

| Mode | Faithfulness | Relevancy | Tradeoff |
|------|--------------|-----------|----------|
| Standard | 94.5% | 82.8% | Balanced |
| Faithful (`fast`) | 95.6% | 82.9% | +1% faith, similar relevancy |
| Confident (0.75 threshold) | **98.1%** | 91.1% | +4% faith, +8% relevancy* |
| Confident (0.80 threshold) | **100%** | 98.3% | +6% faith, +16% relevancy* |

*Higher relevancy because the system **abstains** on uncertain questions. Answered questions are more reliable.

### Strategy 1: Claim Verification (Faithful Mode)

Verifies every claim in the answer against retrieved context. Removes unsupported claims.

```python
rag = RAGSystem(faithful_mode="fast")  # 3 LLM calls per query
# or
rag = RAGSystem(faithful_mode="full")  # N+3 LLM calls (N = number of claims)
```

**Pipeline:**
```
Question → Generate Answer → Extract Claims → Verify Each Claim → Rewrite Answer
```

**Best for:** Domains where you must answer every question but cannot hallucinate.

**Cost:** ~3x latency, ~3x LLM cost per query.

### Strategy 2: Confidence-Gated RAG (Abstention)

Computes a confidence score and **refuses to answer** when uncertain. Better to say "I don't know" than to hallucinate.

```python
rag = RAGSystem(faithful_mode="confident-lite", k=10)  # Recommended
# or
rag = RAGSystem(faithful_mode="confident", k=10)  # Full confidence checks
```

**Confidence signals:**
- **Faithfulness score** (40%): What fraction of claims are verifiable?
- **Context coverage** (30%): Does retrieved context address the question?
- **Source agreement** (20%): Do sources contradict each other?
- **Scope check** (10%): Is the question within the document domain?

**Usage in code:**
```python
from confident_rag import ConfidentRAGModuleLite
from retriever import HybridRetriever

retriever = HybridRetriever()
rag = ConfidentRAGModuleLite(
    retriever,
    k=10,  # Retrieve more for better coverage
    confidence_threshold=0.75,  # Tune this (see below)
)

response = rag("What are the contraindications for this drug?")

if response.abstained:
    print(f"Cannot answer reliably: {response.abstention_reason}")
else:
    print(f"Answer ({response.confidence_score:.0%} confident): {response.answer}")
```

### Tuning the Confidence Threshold

The threshold controls the abstention rate. Higher threshold = fewer answers but higher reliability.

| Threshold | Faithfulness | Relevancy | Abstention Rate | Recommendation |
|-----------|--------------|-----------|-----------------|----------------|
| 0.70 | ~95% | ~85% | ~10% | General use |
| **0.75** | **98%** | **91%** | **~20%** | **Medical/Legal (recommended)** |
| 0.78 | ~99% | ~89% | ~35% | High liability |
| 0.80 | 100% | 98% | ~50% | Zero tolerance for error |

**Finding your threshold:**
```bash
# Test different thresholds on your eval set
uv run evaluation.py full -f test_set.json -o results.json --faithful confident-lite

# Manually test specific threshold in Python
rag = ConfidentRAGModuleLite(retriever, confidence_threshold=0.78)
```

### Strategy 3: Combined Approach

For maximum safety, combine retrieval tuning with confidence gating:

```python
from rag_pipeline import RAGSystem

rag = RAGSystem(
    faithful_mode="confident-lite",
    k=10,  # More documents for better coverage
    hybrid_retrieval=True,  # BM25 + Vector
    # Higher BM25 weight for exact medical terminology
)

# In retriever.py, you can adjust:
# bm25_weight=0.5  (up from 0.4)
# vector_weight=0.5
# over_retrieve_factor=5  (up from 4)
```

### Retrieval Tuning for High-Stakes

| Parameter | Default | High-Stakes Setting | Why |
|-----------|---------|---------------------|-----|
| `k` | 5 | 8-10 | More context for better coverage |
| `over_retrieve_factor` | 4 | 5-6 | More candidates for reranking |
| `bm25_weight` | 0.4 | 0.5 | Exact terminology matters (drug names, legal terms) |
| `reranker` | `llm` | `llm` | Keep LLM reranking for precision |

### When to Use Each Strategy

| Scenario | Strategy | Settings |
|----------|----------|----------|
| Medical diagnosis support | Confident (0.75-0.80) | `faithful_mode="confident-lite"` |
| Legal document Q&A | Confident (0.75) | `faithful_mode="confident-lite"` |
| Drug interaction lookup | Confident (0.80) | Zero tolerance, high threshold |
| Financial compliance | Faithful + Confident | `faithful_mode="confident"` |
| Internal knowledge base | Standard or Faithful | `faithful_mode="fast"` |
| Customer support | Standard | `faithful_mode=None` |

### Monitoring in Production

Track these metrics to tune your threshold:

```python
# Log confidence scores
response = rag(question)
log({
    "question": question,
    "confidence": response.confidence_score,
    "abstained": response.abstained,
    "faithfulness": response.confidence_signals.faithfulness_score,
    "coverage": response.confidence_signals.context_coverage,
})

# Adjust threshold based on:
# 1. Abstention rate (target: 15-25% for medical)
# 2. User feedback on wrong answers
# 3. RAGAS evaluation scores
```

### Evaluation for High-Stakes

```bash
# Run evaluation with confident mode
uv run evaluation.py full -f medical_test_set.json -o results.json --faithful confident-lite

# Compare modes
uv run evaluation.py full -f test_set.json -o baseline.json
uv run evaluation.py full -f test_set.json -o faithful.json --faithful fast
uv run evaluation.py full -f test_set.json -o confident.json --faithful confident-lite
```

**Target metrics for production medical RAG:**
- Faithfulness: >97%
- Answer Relevancy: >85%
- Abstention Rate: 15-25%

---

### Rate Limiting

**DeepSeek:** No rate limits (pay-as-you-go). This is the default.

**Cohere (if using `reranker="cohere"`):** Trial tier is 10 req/min. Auto-throttled:
```python
# retriever.py adds ~6.5s delay between rerank calls
if elapsed < 6.5:
    time.sleep(6.5 - elapsed)
```

---

## Configuration

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bm25_weight` | 0.4 | Keyword search weight |
| `vector_weight` | 0.6 | Semantic search weight |
| `over_retrieve_factor` | 4 | Candidates before rerank |
| `vector_threshold` | 0.2 | Min similarity score |
| `reranker` | "llm" | Reranker type (llm/cohere/local/none) |

### Embedding Models

| Model | Dims | Cost | Quality |
|-------|------|------|---------|
| **text-embedding-3-large** | 1536* | ~$0.13/1M | Best |
| text-embedding-3-small | 1536 | ~$0.02/1M | Good |
| all-MiniLM-L6-v2 | 384 | Free | Basic |

*\*Truncated from 3072 for Supabase HNSW compatibility*

### LLM Options

| Provider | Model | Notes |
|----------|-------|-------|
| **DeepSeek** | **deepseek-chat (V3.2)** | **Default.** Fast, 91.7% faithfulness |
| DeepSeek | deepseek-reasoner (V3.2) | 5-10x slower, 86.5% faithfulness (not recommended) |
| Groq | moonshotai/kimi-k2-instruct | Fast but rate limited |
| Gemini | gemini-2.5-flash | Google |

---

## Project Structure

```
DSPy-Supabase-RAG/
├── rag_pipeline.py       # RAGSystem, RAGModule
├── faithful_rag.py       # FaithfulRAG (claim verification)
├── confident_rag.py      # Confidence-gated RAG (abstention)
├── retriever.py          # HybridRetriever, reranking
├── embeddings.py         # EmbeddingGenerator
├── pdf_processor.py      # PDFProcessor (Docling)
├── evaluation.py         # PipelineEvaluator (RAGAS)
├── eval.md               # Evaluation results
├── supabase_schema.sql   # Database schema
├── requirements.txt      # Dependencies
└── test_set.json         # Evaluation set
```

---

# Setup

---

## Quick Start

```bash
# 1. Install
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure
cp .env.example .env  # Add API keys

# 3. Database
# Run supabase_schema.sql in Supabase SQL Editor

# 4. Ingest
uv run rag_pipeline.py ingest your_docs/*.pdf

# 5. Query
uv run rag_pipeline.py interactive
```

---

## API Keys

```env
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
OPENAI_API_KEY=sk-xxx          # Required for embeddings + GPT-4o-mini evaluation
DEEPSEEK_API_KEY=xxx           # Required for LLM generation + reranking
```

| Provider | Purpose | Pricing | Link |
|----------|---------|---------|------|
| **OpenAI** | Embeddings + Evaluation | GPT-4o-mini: ~$0.15/$0.60 per 1M | [platform.openai.com](https://platform.openai.com) |
| **DeepSeek** | Generation + Reranking | $0.28/$0.42 per 1M | [platform.deepseek.com](https://platform.deepseek.com) |
| Supabase | Vector storage | 500MB free | [supabase.com](https://supabase.com) |

---

## Usage

### Python

```python
from rag_pipeline import RAGSystem

rag = RAGSystem()
rag.ingest("document.pdf")

response = rag.query("What are the key findings?")
print(response.answer)
print(response.sources)
```

### CLI

```bash
uv run rag_pipeline.py ingest *.pdf
uv run rag_pipeline.py query "Your question"
uv run rag_pipeline.py interactive
```

---

## Supabase Schema

Run in SQL Editor:

```sql
create extension if not exists vector with schema extensions;

create table documents (
  id bigint primary key generated always as identity,
  content text not null,
  source text,
  section text,
  metadata jsonb default '{}'::jsonb,
  embedding extensions.vector(1536)
);

create index on documents using hnsw (embedding vector_cosine_ops);
alter table documents enable row level security;
create policy "Allow all" on documents for all using (true);
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `SUPABASE_URL not found` | Add to `.env` |
| `DEEPSEEK_API_KEY not found` | Add to `.env` |
| `OPENAI_API_KEY not found` | Required for embeddings and RAGAS evaluation |
| `match_documents not found` | Run schema SQL |
| `dimension mismatch` | Re-run schema, re-ingest |
| `429 (Cohere)` | Only if using `reranker="cohere"`, switch to `llm` |
| Slow evaluation | GPT-4o-mini is fast, ensure API key is valid |

---

## License

MIT
