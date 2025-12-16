# 🚀 DSPy Supabase RAG - Quick Start Guide

Get your RAG pipeline running in under 10 minutes.

## Prerequisites

- Python 3.11+
- A Supabase account (free tier works)
- A Groq API key (free tier: 30 requests/min)

---

## Step 1: Install Dependencies

### Option A: Using UV (Recommended - 10-100x faster than pip)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project
cd DSPy-Supabase-RAG

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Option B: Using pip

```bash
cd DSPy-Supabase-RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option C: Using the project's existing venv

```bash
cd DSPy-Supabase-RAG

# Use parent project's venv
source ../venv/bin/activate

# Install additional dependencies
pip install -r requirements.txt
```

---

## Step 2: Get API Keys

### Groq API Key (Required - for LLM)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / Sign in
3. Go to **API Keys** → **Create API Key**
4. Copy the key

### Supabase Credentials (Required - for vector storage)

1. Go to [supabase.com](https://supabase.com)
2. Create a new project (or use existing)
3. Go to **Settings** → **API**
4. Copy:
   - **Project URL** (looks like `https://xxxxx.supabase.co`)
   - **service_role key** (NOT the anon key)

### OpenAI API Key (Required - for embeddings and RAGAS evaluation)

1. Go to [platform.openai.com](https://platform.openai.com)
2. Go to **API Keys** → **Create new secret key**
3. Copy the key

### Cohere API Key (Optional - for reranking)

1. Go to [dashboard.cohere.com](https://dashboard.cohere.com/api-keys)
2. Create an API key
3. Copy the key (free tier: 20 requests/min)

---

## Step 3: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your keys
nano .env  # or use any editor
```

Your `.env` should look like:

```env
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here
GROQ_API_KEY=gsk_your-groq-key-here
OPENAI_API_KEY=sk-your-openai-key-here  # For embeddings + RAGAS

# Optional (for improved retrieval precision)
COHERE_API_KEY=your-cohere-key-here  # Free tier: 20 req/min
```

---

## Step 4: Set Up Supabase Database

1. Go to your Supabase project
2. Click **SQL Editor** in the sidebar
3. Click **New Query**
4. Paste the contents of `supabase_schema.sql` and click **Run**

Or copy this SQL directly:

```sql
-- Enable the vector extension
create extension if not exists vector with schema extensions;

-- Create the documents table
-- Using 1536 dimensions (Supabase HNSW limit is 2000)
-- Works with: text-embedding-3-small, or text-embedding-3-large with dimensions=1536
create table documents (
  id bigint primary key generated always as identity,
  content text not null,
  source text,
  section text,
  metadata jsonb default '{}'::jsonb,
  embedding extensions.vector(1536)
);

-- Create an HNSW index for fast similarity search
create index on documents using hnsw (embedding vector_cosine_ops);

-- Enable Row Level Security
alter table documents enable row level security;

-- Create a policy to allow all operations
create policy "Allow all" on documents for all using (true);

-- Create the similarity search function
create or replace function match_documents (
  query_embedding extensions.vector(1536),
  match_threshold float default 0.5,
  match_count int default 5
)
returns table (
  id bigint,
  content text,
  source text,
  section text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.content,
    documents.source,
    documents.section,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
```

You should see "Success. No rows returned" - that's correct!

---

## Step 5: Download Sample PDFs

```bash
# Download 2 sample PDFs for testing
uv run download_samples.py

# Or download all 5 samples
uv run download_samples.py --all
```

This downloads research papers to `./sample_pdfs/`:
- `docling_report.pdf` - Docling Technical Report
- `rag_paper.pdf` - Original RAG paper

---

## Step 6: Ingest Documents

```bash
# Ingest the sample PDFs
uv run rag_pipeline.py ingest sample_pdfs/*.pdf
```

You should see:
```
Processing: sample_pdfs/docling_report.pdf
✓ Processed: 12 pages, 5 tables, 15234 chars
Created 24 semantic chunks
Embedded 24 chunks
Total inserted: 24 records
...
✅ Ingested 48 chunks from 2 documents
```

---

## Step 7: Query Your RAG System

### Single Query

```bash
uv run rag_pipeline.py query "What is Docling and what can it do?"
```

### Interactive Mode

```bash
uv run rag_pipeline.py interactive
```

Then ask questions:
```
❓ Question: What is RAG?
💬 Answer: RAG (Retrieval-Augmented Generation) is a technique that...

❓ Question: How does Docling handle tables?
💬 Answer: Docling uses TableFormer to detect and extract table structures...

❓ Question: quit
```

---

## Step 8: Evaluate Performance (Optional)

### Quick Evaluation (No ground truth needed)

```bash
uv run evaluation.py quick -q "What is Docling?" "How does RAG work?"
```

### Full Evaluation with Test Set

Create `test_set.json`:
```json
[
  {
    "question": "What is Docling?",
    "expected_answer": "Docling is a document processing library that converts PDFs to structured data."
  },
  {
    "question": "What is RAG?",
    "expected_answer": "RAG is Retrieval-Augmented Generation, a technique that enhances LLM responses with retrieved context."
  }
]
```

Then run:
```bash
uv run evaluation.py full -f test_set.json -o results.json
```

---

## 🎉 You're Done!

### What You Just Built

```
PDF Documents
     │
     ▼
┌─────────────────────────────────────┐
│  Docling Parser (with OCR)          │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Contextual Chunking                │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  OpenAI Embeddings (1536 dims)      │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Supabase pgvector + BM25 Hybrid    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Cohere Reranker (optional)         │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Groq kimi-k2-instruct              │
└─────────────────────────────────────┘
```

### Next Steps

1. **Add your own PDFs:**
   ```bash
   python rag_pipeline.py ingest /path/to/your/documents/*.pdf
   ```

2. **Use from Python:**
   ```python
   from rag_pipeline import RAGSystem
   
   rag = RAGSystem()
   rag.ingest("my_document.pdf")
   response = rag.query("What is this document about?")
   print(response.answer)
   ```

3. **Optimize with DSPy:**
   Check out the DSPy documentation for prompt optimization with MIPROv2.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'docling'"

```bash
uv pip install docling  # or: pip install docling
```

### "SUPABASE_URL not found"

Make sure you copied `.env.example` to `.env` and filled in your credentials.

### "match_documents function not found"

You need to run the SQL setup in Supabase (Step 4).

### "Rate limit exceeded" from Groq

Groq free tier allows 30 requests/minute. Wait a moment or upgrade your plan.

### Slow queries with reranking

Cohere's free tier allows only 20 requests/minute. The retriever automatically throttles to stay within limits, adding ~6 second delays between rerank calls. To speed up:
- Upgrade your Cohere plan, or
- Disable reranking: `HybridRetriever(use_reranker=False)`

### OCR not working on scanned PDFs

```bash
uv pip install easyocr  # or: pip install easyocr
```

### Slow first run

The first time you run, it downloads:
- Sentence transformer model (~90MB)
- Docling models (~500MB)

Subsequent runs are fast.

---

## Project Structure

```
DSPy-Supabase-RAG/
├── pdf_processor.py      # Docling PDF parsing + OCR
├── embeddings.py         # Embedding generation + Supabase storage
├── retriever.py          # Hybrid search (BM25 + vector)
├── rag_pipeline.py       # Main RAG system
├── evaluation.py         # RAGAS + LLM-as-Judge evaluation
├── download_samples.py   # Download test PDFs
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
├── .env                  # Your credentials (don't commit!)
├── README.md             # Full documentation
└── START_HERE.md         # This file
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python download_samples.py` | Download sample PDFs |
| `python rag_pipeline.py ingest FILE` | Ingest a document |
| `python rag_pipeline.py query "..."` | Ask a question |
| `python rag_pipeline.py interactive` | Interactive Q&A mode |
| `python evaluation.py quick -q "..."` | Quick evaluation |
| `python evaluation.py full -f FILE` | Full evaluation |
| `python pdf_processor.py FILE` | Process PDF only |
| `python embeddings.py search "..."` | Search embeddings |
| `python retriever.py "..."` | Test retrieval |

