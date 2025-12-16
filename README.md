# DSPy Supabase RAG Pipeline

A production-ready Retrieval-Augmented Generation system with hybrid search, neural reranking, and comprehensive evaluation.

```
Documents → Semantic Chunks → Vector Embeddings → Hybrid Retrieval → Reranking → LLM Generation
```

---

## Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Answer Relevancy** | **93.6%** | Generated answers address the question |
| **Context Precision** | **72.5%** | Retrieved chunks are relevant |
| **Context Recall** | 64.6% | Coverage of relevant information |
| **Faithfulness** | 73.0% | Answers grounded in context |

*Answer Relevancy is typically the most important metric—it measures whether users get useful answers.*

*Evaluated on 20 samples using [RAGAS](https://docs.ragas.io/) with GPT-4o-mini as judge*

---

## Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **PDF Parsing** | [Docling](https://github.com/DS4SD/docling) | Layout analysis, table extraction, OCR |
| **Embeddings** | OpenAI text-embedding-3-large | 3072→1536 dims (Supabase HNSW limit) |
| **Vector Store** | Supabase pgvector | PostgreSQL + HNSW index |
| **Keyword Search** | rank-bm25 | BM25Okapi algorithm |
| **Reranking** | Cohere rerank-v3.5 | Cross-encoder neural reranker |
| **LLM** | Groq (kimi-k2-instruct) | Fast inference |
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
                  │  Cohere Reranker    │  ← +10% precision
                  │  (rerank-v3.5)      │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │     Groq LLM        │
                  │ (kimi-k2-instruct)  │
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
| Rerank | Cohere rerank-v3.5 | Keep top 5 (precision) |

**Impact:** Context precision 62.7% → 72.5% (+10%)

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

### Running Evaluation

```bash
uv run evaluation.py full -f test_set.json -o results.json

# With rate limit delay
uv run evaluation.py full -f test_set.json -o results.json --delay 5
```

**Note:** RAGAS uses LLM-as-judge with ~10% variance. Use 50+ samples for stable metrics.

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

### Rate Limiting

Cohere trial: **10 req/min**. Auto-throttled in code:

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
| `use_reranker` | True | Enable Cohere reranking |

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
| **Groq** | moonshotai/kimi-k2-instruct | Default |
| Groq | llama-3.3-70b-versatile | Alternative |
| Gemini | gemini-2.5-flash | Google |

---

## Project Structure

```
DSPy-Supabase-RAG/
├── rag_pipeline.py       # RAGSystem, RAGModule
├── retriever.py          # HybridRetriever, reranking
├── embeddings.py         # EmbeddingGenerator
├── pdf_processor.py      # PDFProcessor (Docling)
├── evaluation.py         # PipelineEvaluator (RAGAS)
├── supabase_schema.sql   # Database schema
├── requirements.txt      # Dependencies
└── test_set.json         # Evaluation set
```

---

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
OPENAI_API_KEY=sk-xxx
GROQ_API_KEY=gsk_xxx
COHERE_API_KEY=xxx  # Optional, +10% precision
```

| Provider | Free Tier | Link |
|----------|-----------|------|
| Supabase | 500MB | [supabase.com](https://supabase.com) |
| OpenAI | Pay-as-you-go | [platform.openai.com](https://platform.openai.com) |
| Groq | 30 req/min | [console.groq.com](https://console.groq.com) |
| Cohere | 10 req/min | [dashboard.cohere.com](https://dashboard.cohere.com) |

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
| `match_documents not found` | Run schema SQL |
| `dimension mismatch` | Re-run schema, re-ingest |
| `429 (Cohere)` | Trial: 10/min, auto-throttled |
| `429 (Groq)` | Add `--delay 5` |
| Slow reranking | Expected ~6.5s/query on trial |

---

## License

MIT
