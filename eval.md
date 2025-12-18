# Evaluation Results

---

## TL;DR

**Use confident-lite mode for medical/legal RAG.** 98%+ faithfulness with 96% answer relevancy.

| Mode | Faithfulness | Answer Relevancy | Use Case |
|------|--------------|------------------|----------|
| **Confident-Lite** | **98.3%** | **96.0%** | Medical/Legal (recommended) |
| Standard | 97.1% | 58.5% | General use |

---

## Medical Paper Evaluation (Latest)

*15 questions on 6 arXiv medical AI papers, evaluated with GPT-4o-mini as judge*

### Test Corpus

| Paper | Topic | Chunks |
|-------|-------|--------|
| chest_xray_diagnosis.pdf | CheXNet, 121-layer CNN for X-ray diagnosis | 32 |
| clinical_bert.pdf | ClinicalBERT for clinical NLP | 47 |
| covid_forecasting.pdf | COVID-19 prediction models | 23 |
| drug_discovery_ai.pdf | AdvProp for molecular property prediction | 81 |
| medical_imaging_ai.pdf | AI in medical imaging review | 123 |
| skin_cancer_classification.pdf | ISIC Challenge, skin lesion analysis | 28 |

**Total: 334 chunks from 6 papers**

### Results (Confident-Lite Mode)

| Metric | Score |
|--------|-------|
| **Faithfulness** | **98.3%** |
| **Answer Relevancy** | **96.0%** |
| **Context Precision** | 93.6% |
| **Context Recall** | 93.3% |
| **Overall** | **95.3%** |

### Configuration

```python
rag = RAGSystem(faithful_mode="confident-lite")
# confidence_threshold=0.60
# k=10 (retrieve more for medical coverage)
```

---

## Original Benchmark (RAG/Docling Papers)

*20 samples, evaluated with GPT-4o-mini as judge*

| Model | Answer Relevancy | Faithfulness | Latency | Verdict |
|-------|------------------|--------------|---------|---------|
| **deepseek-chat** | **92.5%** | **91.7%** | Fast | **Recommended** |
| deepseek-reasoner | 93.0% | 86.5% | 5-10x slower | Not worth it |

---

## Chat vs Reasoner

### DeepSeek-Chat (V3.2)

| Metric | Score |
|--------|-------|
| **Answer Relevancy** | 92.5% |
| **Faithfulness** | 91.7% |
| Context Precision | 76.9% |
| Context Recall | 70.4% |
| **Overall** | 82.9% |

- Fast inference
- Both key metrics >90%
- Conservative answers that stick to context

### DeepSeek-Reasoner (V3.2)

| Metric | Normal | Faithful |
|--------|--------|----------|
| **Answer Relevancy** | 93.0% | 78.6% |
| **Faithfulness** | 86.5% | 89.2% |
| Context Precision | 77.0% | 73.5% |
| Context Recall | 63.3% | 68.7% |
| **Overall** | 80.0% | 77.5% |

- 5-10x slower due to chain-of-thought
- **Lower faithfulness than chat** even in normal mode (86.5% vs 91.7%)
- Faithful mode barely helps (+2.7% faithfulness) but tanks relevancy (-14.4%)
- Not recommended for RAG

---

## Faithful Mode (Claim Verification)

Optional post-processing that verifies claims against context.

| Metric | Normal | Faithful | Delta |
|--------|--------|----------|-------|
| **Faithfulness** | 91.7% | **95.6%** | +3.9% |
| **Answer Relevancy** | **92.5%** | 82.9% | -9.6% |
| Overall | 82.9% | 79.3% | -3.6% |

**Trade-off:** +4% faithfulness costs -10% relevancy.

**When to use:** High-stakes domains (medical, legal, financial) where hallucination risk > completeness.

**When to skip:** General use. Normal mode already exceeds 90% on both metrics.

---

## Pipeline Configuration

| Component | Value |
|-----------|-------|
| Generator | DeepSeek (deepseek-chat) |
| Reranker | DeepSeek LLM |
| Evaluator | GPT-4o-mini (different model to avoid self-eval bias) |
| Embeddings | text-embedding-3-large (1536 dims) |
| Retrieval | Hybrid (BM25 + Vector + RRF) |

### DeepSeek Pricing (Chat & Reasoner)

| Type | Cost |
|------|------|
| Input (cache hit) | $0.028 / 1M tokens |
| Input (cache miss) | $0.28 / 1M tokens |
| Output | $0.42 / 1M tokens |

---

## Score Interpretation

| Range | Quality | Action |
|-------|---------|--------|
| > 80% | Excellent | Production ready |
| 60-80% | Good | Minor tuning |
| 40-60% | Fair | Review retrieval |
| < 40% | Poor | Debug pipeline |

---

## Per-Question Breakdown (Chat Normal)

| # | Question | Faith. | Relev. |
|---|----------|--------|--------|
| 1 | What is Docling and what is its primary purpose? | 100% | 90% |
| 2 | What document formats does Docling support? | 100% | 95% |
| 3 | How does Docling handle tables in documents? | 100% | 100% |
| 4 | What OCR capabilities does Docling provide? | 91% | 98% |
| 5 | What is RAG and why was it developed? | 100% | 93% |
| 6 | What are the two main components of the RAG architecture? | 100% | 100% |
| 7 | How does RAG reduce hallucinations in language models? | 100% | 100% |
| 8 | What is Dense Passage Retrieval (DPR) in the context of RAG? | 100% | 100% |
| 9 | What types of knowledge-intensive tasks does RAG excel at? | 100% | 95% |
| 10 | What is the difference between RAG-Sequence and RAG-Token models? | 100% | 98% |
| 11 | How does Docling handle document layout analysis? | 100% | 99% |
| 12 | What output formats can Docling generate? | 100% | 98% |
| 13 | Why is chunking important in RAG pipelines? | 100% | 96% |
| 14 | What advantages does RAG have over fine-tuning? | 75% | 98% |
| 15 | How does the retrieval component in RAG find relevant documents? | 100% | 100% |
| 16 | What role does the embedding model play in a RAG system? | 36% | 100% |
| 17 | How does Docling extract text from scanned PDF documents? | 83% | 96% |
| 18 | What is contextual chunking and why is it useful? | 50% | 95% |
| 19 | What benchmarks were used to evaluate RAG in the original paper? | 100% | 98% |
| 20 | How does hybrid search improve RAG retrieval? | 94% | 0% |

### Low Performers

**Faithfulness < 90%:**
- Q14 (75%): Answer extrapolated advantages not in context
- Q16 (36%): Embedding model role elaborated beyond retrieved docs
- Q17 (83%): Minor OCR details not fully grounded
- Q18 (50%): Contextual chunking definition not in context

**Relevancy = 0%:**
- Q20: "Hybrid search" not covered in retrieved context (retrieval gap)
