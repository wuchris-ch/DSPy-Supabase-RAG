# RAG Pipeline Evaluation Report

Generated: 2025-12-17T13:08:12.825318

## Summary

| Metric | Score |
|--------|-------|
| **Overall** | 77.5% |
| Context Precision | 73.5% |
| Context Recall | 68.7% |
| Context Relevance | 0.0% |
| Faithfulness | 89.2% |
| Answer Relevancy | 78.6% |
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
