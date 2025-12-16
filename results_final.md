# RAG Pipeline Evaluation Report

Generated: 2025-12-15T17:52:44.105304

## Summary

| Metric | Score |
|--------|-------|
| **Overall** | 72.1% |
| Context Precision | 66.9% |
| Context Recall | 60.4% |
| Context Relevance | 0.0% |
| Faithfulness | 71.8% |
| Answer Relevancy | 89.3% |
| Answer Correctness | 0.0% |

## Configuration

```json
{
  "evaluator": "ragas",
  "model": "gpt-4o-mini"
}
```

## Samples Evaluated: 20

### Interpretation Guide

- **Overall > 80%**: Excellent - production ready
- **Overall 60-80%**: Good - consider optimization
- **Overall 40-60%**: Fair - needs improvement
- **Overall < 40%**: Poor - major issues to address

### Metric Explanations

- **Context Precision**: Are the retrieved documents relevant?
- **Context Recall**: Did we retrieve all relevant information?
- **Context Relevance**: Overall quality of retrieved context
- **Faithfulness**: Is the answer grounded in the context (no hallucination)?
- **Answer Relevancy**: Does the answer address the question?
- **Answer Correctness**: Is the answer factually correct?
