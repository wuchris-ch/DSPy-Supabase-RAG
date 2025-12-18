"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RAG Evaluation - Comprehensive Pipeline Assessment (2024/2025)              ║
║                                                                              ║
║  Multiple evaluation approaches:                                             ║
║  1. RAGAS - Industry standard RAG metrics                                    ║
║  2. DSPy SemanticF1 - Built-in DSPy evaluation                              ║
║  3. LLM-as-Judge - Custom LLM-based evaluation                              ║
║  4. Component-level metrics - Retrieval + Generation separately             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Based on latest research and best practices:
- RAGAS paper: https://arxiv.org/abs/2309.15217
- RAGProbe: https://arxiv.org/abs/2409.19019
- DSPy Evaluation: https://dspy.ai/tutorials/rag
"""

import os
import json
import logging
import time
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import dspy
from dotenv import load_dotenv

# Optional MLflow integration
try:
    from mlflow_tracking import MLflowTracker, MLflowConfig
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    ground_truth: str  # Expected answer (reference)
    contexts: list[str] = field(default_factory=list)  # Retrieved contexts
    answer: str = ""  # Generated answer
    metadata: dict = field(default_factory=dict)
    abstained: bool = False  # Whether system refused to answer
    abstention_reason: str = ""  # Why it abstained
    confidence_score: float = 1.0  # Confidence if using confident mode


@dataclass
class ComponentScores:
    """Scores broken down by RAG component."""
    # Retrieval metrics
    context_precision: float = 0.0  # Are retrieved docs relevant?
    context_recall: float = 0.0  # Did we get all relevant docs?
    context_relevance: float = 0.0  # Overall context quality

    # Generation metrics
    faithfulness: float = 0.0  # Is answer grounded in context?
    answer_relevancy: float = 0.0  # Does answer address the question?
    answer_correctness: float = 0.0  # Is answer factually correct?

    # Abstention metrics (for confident mode)
    abstention_rate: float = 0.0  # How often system refused to answer
    answered_faithfulness: float = 0.0  # Faithfulness only for non-abstained
    avg_confidence: float = 0.0  # Average confidence when answering

    # Combined
    overall: float = 0.0


@dataclass
class EvalResult:
    """Complete evaluation results."""
    scores: ComponentScores
    num_samples: int
    samples_detail: list[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        abstention_section = ""
        if self.scores.abstention_rate > 0:
            abstention_section = f"""╠══════════════════════════════════════════════════════════════╣
║  ABSTENTION METRICS                                          ║
║  ├─ Abstention Rate:   {self.scores.abstention_rate:>6.1%}                            ║
║  ├─ Answered Faith.:   {self.scores.answered_faithfulness:>6.1%}                            ║
║  └─ Avg Confidence:    {self.scores.avg_confidence:>6.1%}                            ║
"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  RAG Pipeline Evaluation Results                             ║
╠══════════════════════════════════════════════════════════════╣
║  Overall Score:        {self.scores.overall:>6.1%}                            ║
╠══════════════════════════════════════════════════════════════╣
║  RETRIEVAL METRICS                                           ║
║  ├─ Context Precision: {self.scores.context_precision:>6.1%}                            ║
║  ├─ Context Recall:    {self.scores.context_recall:>6.1%}                            ║
║  └─ Context Relevance: {self.scores.context_relevance:>6.1%}                            ║
╠══════════════════════════════════════════════════════════════╣
║  GENERATION METRICS                                          ║
║  ├─ Faithfulness:      {self.scores.faithfulness:>6.1%}                            ║
║  ├─ Answer Relevancy:  {self.scores.answer_relevancy:>6.1%}                            ║
║  └─ Answer Correctness:{self.scores.answer_correctness:>6.1%}                            ║
{abstention_section}╠══════════════════════════════════════════════════════════════╣
║  Samples Evaluated: {self.num_samples:<5}                                   ║
║  Timestamp: {self.timestamp[:19]:<20}                        ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    def to_dict(self) -> dict:
        return {
            "scores": {
                "overall": self.scores.overall,
                "retrieval": {
                    "context_precision": self.scores.context_precision,
                    "context_recall": self.scores.context_recall,
                    "context_relevance": self.scores.context_relevance,
                },
                "generation": {
                    "faithfulness": self.scores.faithfulness,
                    "answer_relevancy": self.scores.answer_relevancy,
                    "answer_correctness": self.scores.answer_correctness,
                },
            },
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
            "config": self.config,
            "samples_detail": self.samples_detail,
        }
    
    def save(self, path: str | Path):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS Evaluator (Industry Standard)
# ═══════════════════════════════════════════════════════════════════════════════

class RAGASEvaluator:
    """
    Evaluate RAG pipelines using RAGAS metrics.

    RAGAS (Retrieval Augmented Generation Assessment) is the industry standard
    for RAG evaluation. Paper: https://arxiv.org/abs/2309.15217

    Metrics:
    ════════

    1. FAITHFULNESS (Generation)
       - Measures if claims in answer are supported by context
       - Extracts claims from answer, verifies each against context
       - Score = (supported claims) / (total claims)

    2. ANSWER RELEVANCY (Generation)
       - Measures if answer addresses the question
       - Generates questions that the answer would be appropriate for
       - Score = similarity between generated questions and original

    3. CONTEXT PRECISION (Retrieval)
       - Measures if retrieved contexts are relevant to question
       - Ranks contexts by relevance, penalizes irrelevant ones at top
       - Uses LLM to judge relevance of each context

    4. CONTEXT RECALL (Retrieval)
       - Measures if all ground truth info was retrieved
       - Extracts claims from ground truth, checks if contexts cover them
       - Requires ground truth answer

    Note: Uses a DIFFERENT model than the generator to avoid self-evaluation bias.
    Default: gpt-5-mini with high reasoning for accurate evaluation.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,  # Only for GPT-5 models
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            model: OpenAI model for evaluation (default: gpt-5-mini)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            reasoning_effort: Reasoning level for GPT-5 models (minimal/low/medium/high)
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.reasoning_effort = reasoning_effort

        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not found! RAGAS evaluation requires OpenAI.\n"
                "Set OPENAI_API_KEY in your .env file."
            )

        self._initialized = False
    
    def _init_ragas(self):
        """Lazy initialization of RAGAS components."""
        if self._initialized:
            return
        
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI
            from datasets import Dataset
            
            self._evaluate = evaluate
            self._Dataset = Dataset
            
            # Initialize LLM with reasoning effort for GPT-5 models
            model_kwargs = {}
            if self.model.startswith("gpt-5") and self.reasoning_effort:
                model_kwargs["reasoning_effort"] = self.reasoning_effort
                logger.info(f"Using {self.model} with reasoning_effort={self.reasoning_effort}")

            llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                model_kwargs=model_kwargs,
            )
            evaluator_llm = LangchainLLMWrapper(llm)
            
            # Configure metrics with LLM
            self._faithfulness = faithfulness
            self._answer_relevancy = answer_relevancy
            self._context_precision = context_precision
            self._context_recall = context_recall
            
            # Set LLM for each metric
            for metric in [self._faithfulness, self._answer_relevancy, 
                          self._context_precision, self._context_recall]:
                metric.llm = evaluator_llm
            
            self._initialized = True
            logger.info("RAGAS initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"RAGAS dependencies not installed: {e}\n"
                "Install with: pip install ragas langchain-openai"
            )
    
    def evaluate(
        self,
        samples: list[EvalSample],
        metrics: Optional[list[str]] = None,
    ) -> EvalResult:
        """
        Evaluate samples using RAGAS metrics.
        
        Args:
            samples: List of EvalSample objects
            metrics: Optional list of metric names to compute
                     ["faithfulness", "answer_relevancy", 
                      "context_precision", "context_recall"]
                      
        Returns:
            EvalResult with all scores
        """
        self._init_ragas()
        
        # Prepare data for RAGAS format
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }
        
        dataset = self._Dataset.from_dict(data)
        
        # Select metrics
        metric_map = {
            "faithfulness": self._faithfulness,
            "answer_relevancy": self._answer_relevancy,
            "context_precision": self._context_precision,
            "context_recall": self._context_recall,
        }
        
        if metrics:
            selected_metrics = [metric_map[m] for m in metrics if m in metric_map]
        else:
            selected_metrics = list(metric_map.values())
        
        # Run evaluation
        logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")
        result = self._evaluate(dataset=dataset, metrics=selected_metrics)
        
        # Extract scores from RAGAS EvaluationResult
        # In RAGAS 0.2.x+, scores are in a pandas DataFrame via to_pandas()
        def get_score(result_obj, metric_name: str) -> float:
            """Safely extract score from RAGAS EvaluationResult."""
            # Method 1: Try to_pandas() and get mean (RAGAS 0.2.x+)
            try:
                df = result_obj.to_pandas()
                if metric_name in df.columns:
                    score = df[metric_name].mean()
                    if not pd.isna(score):
                        logger.debug(f"Got {metric_name}={score} from to_pandas()")
                        return float(score)
            except Exception as e:
                logger.debug(f"to_pandas() failed for {metric_name}: {e}")
            
            # Method 2: Try direct subscript access (older RAGAS)
            try:
                score = result_obj[metric_name]
                if score is not None:
                    logger.debug(f"Got {metric_name}={score} from subscript")
                    return float(score)
            except (KeyError, TypeError) as e:
                logger.debug(f"Subscript access failed for {metric_name}: {e}")
            
            # Method 3: Try attribute access
            try:
                score = getattr(result_obj, metric_name, None)
                if score is not None:
                    logger.debug(f"Got {metric_name}={score} from attribute")
                    return float(score)
            except (AttributeError, TypeError) as e:
                logger.debug(f"Attribute access failed for {metric_name}: {e}")
            
            logger.warning(f"Could not extract score for {metric_name}, returning 0.0")
            return 0.0
        
        # Import pandas for score extraction
        import pandas as pd
        
        scores = ComponentScores(
            faithfulness=get_score(result, "faithfulness"),
            answer_relevancy=get_score(result, "answer_relevancy"),
            context_precision=get_score(result, "context_precision"),
            context_recall=get_score(result, "context_recall"),
        )
        
        logger.info(f"Extracted scores: faithfulness={scores.faithfulness:.3f}, "
                   f"answer_relevancy={scores.answer_relevancy:.3f}, "
                   f"context_precision={scores.context_precision:.3f}, "
                   f"context_recall={scores.context_recall:.3f}")
        
        # Calculate overall score (average of all metrics)
        valid_scores = [s for s in [
            scores.faithfulness, scores.answer_relevancy,
            scores.context_precision, scores.context_recall
        ] if s > 0]
        scores.overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Extract per-sample details
        samples_detail = []
        try:
            df = result.to_pandas()
            for i, sample in enumerate(samples):
                detail = {
                    "question": sample.question,
                    "answer": sample.answer[:200] + "..." if len(sample.answer) > 200 else sample.answer,
                }
                # Add individual scores if available
                if i < len(df):
                    row = df.iloc[i]
                    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        if metric in df.columns:
                            val = row[metric]
                            detail[metric] = float(val) if not pd.isna(val) else None
                samples_detail.append(detail)
        except Exception as e:
            logger.warning(f"Could not extract sample details: {e}")
        
        return EvalResult(
            scores=scores,
            num_samples=len(samples),
            samples_detail=samples_detail,
            config={"evaluator": "ragas", "model": self.model},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy Evaluator (Built-in)
# ═══════════════════════════════════════════════════════════════════════════════

class DSPyEvaluator:
    """
    Evaluate RAG using DSPy's built-in SemanticF1 metric.
    
    SemanticF1 works by:
    1. Decomposing both prediction and ground truth into atomic facts
    2. Computing precision: what % of predicted facts are in ground truth
    3. Computing recall: what % of ground truth facts are in prediction
    4. Returning F1 score (harmonic mean)
    
    Advantages:
    - Uses your configured LLM (Groq, Gemini, etc.)
    - No separate API key needed
    - Integrated with DSPy optimization
    """
    
    def __init__(self):
        """Initialize DSPy evaluator."""
        self._metric = None
    
    def _init_metric(self):
        """Lazy initialization of SemanticF1 metric."""
        if self._metric is None:
            from dspy.evaluate import SemanticF1
            self._metric = SemanticF1(decompositional=True)
    
    def evaluate_single(
        self,
        prediction: str,
        ground_truth: str,
    ) -> float:
        """
        Evaluate a single prediction against ground truth.
        
        Returns:
            SemanticF1 score (0-1)
        """
        self._init_metric()
        
        # Create DSPy examples
        gold = dspy.Example(response=ground_truth)
        pred = dspy.Example(response=prediction)
        
        score = self._metric(gold, pred)
        return float(score)
    
    def evaluate_rag_module(
        self,
        rag_module,
        test_set: list[dict],
        verbose: bool = True,
    ) -> EvalResult:
        """
        Evaluate a DSPy RAG module on a test set.
        
        Args:
            rag_module: DSPy module with forward(question) method
            test_set: List of {"question": ..., "expected_answer": ...}
            verbose: Print progress
            
        Returns:
            EvalResult with scores
        """
        self._init_metric()
        
        scores = []
        details = []
        
        for i, item in enumerate(test_set):
            question = item["question"]
            expected = item["expected_answer"]
            
            # Get prediction
            try:
                result = rag_module(question)
                answer = result.answer if hasattr(result, 'answer') else str(result)
            except Exception as e:
                logger.warning(f"Error on sample {i}: {e}")
                answer = ""
            
            # Compute score
            score = self.evaluate_single(answer, expected)
            scores.append(score)
            
            details.append({
                "question": question,
                "expected": expected,
                "answer": answer,
                "score": score,
            })
            
            if verbose:
                status = "✓" if score > 0.5 else "✗"
                print(f"  [{status}] Q{i+1}: {question[:40]}... → {score:.2f}")
        
        # Calculate aggregate scores
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        component_scores = ComponentScores(
            answer_correctness=avg_score,
            overall=avg_score,
        )
        
        return EvalResult(
            scores=component_scores,
            num_samples=len(test_set),
            samples_detail=details,
            config={"evaluator": "dspy_semantic_f1"},
        )
    
    def evaluate_with_dspy_evaluate(
        self,
        rag_module,
        devset: list,
        num_threads: int = 8,
    ) -> EvalResult:
        """
        Use DSPy's built-in Evaluate class for parallel evaluation.
        
        Args:
            rag_module: DSPy module to evaluate
            devset: List of dspy.Example objects
            num_threads: Number of parallel threads
            
        Returns:
            EvalResult with scores
        """
        self._init_metric()
        
        # Create evaluator
        evaluator = dspy.Evaluate(
            devset=devset,
            metric=self._metric,
            num_threads=num_threads,
            display_progress=True,
            display_table=5,
        )
        
        # Run evaluation
        result = evaluator(rag_module)
        
        component_scores = ComponentScores(
            answer_correctness=result.score / 100 if result.score else 0.0,
            overall=result.score / 100 if result.score else 0.0,
        )
        
        return EvalResult(
            scores=component_scores,
            num_samples=len(devset),
            config={"evaluator": "dspy_evaluate"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-as-Judge Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class JudgeFaithfulness(dspy.Signature):
    """Judge if the answer is faithful to the provided context.
    Score from 0-10 where 10 means completely faithful (no hallucination)."""
    
    context = dspy.InputField(desc="The retrieved context passages")
    answer = dspy.InputField(desc="The generated answer")
    
    reasoning = dspy.OutputField(desc="Explanation of faithfulness assessment")
    score = dspy.OutputField(desc="Score from 0-10")


class JudgeRelevancy(dspy.Signature):
    """Judge if the answer is relevant to the question.
    Score from 0-10 where 10 means perfectly relevant and complete."""
    
    question = dspy.InputField(desc="The user's question")
    answer = dspy.InputField(desc="The generated answer")
    
    reasoning = dspy.OutputField(desc="Explanation of relevancy assessment")
    score = dspy.OutputField(desc="Score from 0-10")


class JudgeContextQuality(dspy.Signature):
    """Judge if the retrieved contexts are relevant to the question.
    Score from 0-10 where 10 means all contexts are highly relevant."""
    
    question = dspy.InputField(desc="The user's question")
    contexts = dspy.InputField(desc="The retrieved context passages")
    
    reasoning = dspy.OutputField(desc="Explanation of context quality")
    score = dspy.OutputField(desc="Score from 0-10")


class LLMJudgeEvaluator:
    """
    LLM-as-Judge evaluation using your configured LLM.
    
    This approach uses the LLM itself to judge quality on multiple dimensions.
    Works with any LLM configured in DSPy (Groq, Gemini, etc.).
    
    Advantages:
    - No separate OpenAI key needed
    - Customizable judging criteria
    - Provides reasoning for scores
    
    Metrics:
    - Faithfulness: Is answer grounded in context?
    - Relevancy: Does answer address the question?
    - Context Quality: Are retrieved contexts relevant?
    """
    
    def __init__(self):
        """Initialize LLM judge evaluator."""
        self.judge_faithfulness = dspy.ChainOfThought(JudgeFaithfulness)
        self.judge_relevancy = dspy.ChainOfThought(JudgeRelevancy)
        self.judge_context = dspy.ChainOfThought(JudgeContextQuality)
    
    def _parse_score(self, score_str: str) -> float:
        """Parse score from LLM output, handling various formats."""
        try:
            # Try to extract number from string
            import re
            numbers = re.findall(r'[\d.]+', str(score_str))
            if numbers:
                score = float(numbers[0])
                return min(10, max(0, score)) / 10  # Normalize to 0-1
            return 0.0
        except:
            return 0.0
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        """
        Evaluate a single RAG response.
        
        Returns:
            Dict with scores and reasoning for each metric
        """
        context_str = "\n\n".join(contexts)
        
        # Judge faithfulness
        faith_result = self.judge_faithfulness(
            context=context_str,
            answer=answer,
        )
        
        # Judge relevancy
        rel_result = self.judge_relevancy(
            question=question,
            answer=answer,
        )
        
        # Judge context quality
        ctx_result = self.judge_context(
            question=question,
            contexts=context_str,
        )
        
        return {
            "faithfulness": {
                "score": self._parse_score(faith_result.score),
                "reasoning": faith_result.reasoning,
            },
            "relevancy": {
                "score": self._parse_score(rel_result.score),
                "reasoning": rel_result.reasoning,
            },
            "context_quality": {
                "score": self._parse_score(ctx_result.score),
                "reasoning": ctx_result.reasoning,
            },
        }
    
    def evaluate(
        self,
        samples: list[EvalSample],
        verbose: bool = True,
    ) -> EvalResult:
        """
        Evaluate multiple samples using LLM-as-Judge.
        
        Args:
            samples: List of EvalSample objects
            verbose: Print progress
            
        Returns:
            EvalResult with scores
        """
        all_faithfulness = []
        all_relevancy = []
        all_context = []
        details = []
        
        for i, sample in enumerate(samples):
            if verbose:
                print(f"  Evaluating sample {i+1}/{len(samples)}...")
            
            result = self.evaluate_single(
                question=sample.question,
                answer=sample.answer,
                contexts=sample.contexts,
            )
            
            all_faithfulness.append(result["faithfulness"]["score"])
            all_relevancy.append(result["relevancy"]["score"])
            all_context.append(result["context_quality"]["score"])
            
            details.append({
                "question": sample.question,
                "answer": sample.answer[:100] + "...",
                **result,
            })
        
        # Calculate averages
        avg_faith = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
        avg_rel = sum(all_relevancy) / len(all_relevancy) if all_relevancy else 0
        avg_ctx = sum(all_context) / len(all_context) if all_context else 0
        
        scores = ComponentScores(
            faithfulness=avg_faith,
            answer_relevancy=avg_rel,
            context_relevance=avg_ctx,
            overall=(avg_faith + avg_rel + avg_ctx) / 3,
        )
        
        return EvalResult(
            scores=scores,
            num_samples=len(samples),
            samples_detail=details,
            config={"evaluator": "llm_judge"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Evaluator (All-in-One)
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineEvaluator:
    """
    Comprehensive RAG pipeline evaluation combining multiple approaches.

    Usage:
        evaluator = PipelineEvaluator(rag_system)

        # Quick evaluation (uses LLM-as-Judge)
        result = evaluator.quick_eval(test_questions)

        # Full evaluation (uses RAGAS if OpenAI key available)
        result = evaluator.full_eval(test_set)

        # Generate evaluation report
        evaluator.generate_report(result, "eval_report.json")

        # With MLflow tracking
        evaluator = PipelineEvaluator(rag_system, enable_mlflow=True)
        result = evaluator.full_eval(test_set)  # Automatically logged to MLflow
    """

    def __init__(
        self,
        rag_system=None,
        enable_mlflow: bool = False,
        mlflow_config=None,
        evaluator_model: str = "gpt-4o-mini",
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize pipeline evaluator.

        Args:
            rag_system: Optional RAG system to evaluate
            enable_mlflow: Enable MLflow tracking for evaluations
            mlflow_config: Optional MLflowConfig for custom settings
            evaluator_model: Model for RAGAS evaluation (default: gpt-4o-mini)
            reasoning_effort: Reasoning level for GPT-5 models (optional)
        """
        self.rag_system = rag_system
        self.evaluator_model = evaluator_model
        self.reasoning_effort = reasoning_effort
        self._dspy_eval = DSPyEvaluator()
        self._llm_judge = LLMJudgeEvaluator()
        self._ragas_eval = None

        # MLflow tracking
        self._mlflow_tracker = None
        if enable_mlflow:
            if not MLFLOW_AVAILABLE:
                logger.warning("MLflow not available. Install with: pip install mlflow")
            else:
                self._mlflow_tracker = MLflowTracker(mlflow_config)
                logger.info(f"MLflow tracking enabled: {self._mlflow_tracker.get_experiment_url()}")
    
    def _get_ragas(self) -> Optional[RAGASEvaluator]:
        """Get RAGAS evaluator if OpenAI key available."""
        if self._ragas_eval is None:
            try:
                self._ragas_eval = RAGASEvaluator(
                    model=self.evaluator_model,
                    reasoning_effort=self.reasoning_effort,
                )
                if self._ragas_eval.api_key:
                    return self._ragas_eval
            except:
                pass
        return self._ragas_eval if self._ragas_eval and self._ragas_eval.api_key else None
    
    def run_rag(
        self,
        questions: list[str],
        delay_seconds: float = 0,
    ) -> list[EvalSample]:
        """
        Run RAG system on questions and collect samples for evaluation.

        Args:
            questions: List of questions to ask
            delay_seconds: Delay between queries (default 0, not needed for DeepSeek)

        Returns:
            List of EvalSample with answers and contexts
        """
        if self.rag_system is None:
            raise ValueError("RAG system not provided")

        samples = []
        for i, q in enumerate(questions):
            try:
                result = self.rag_system.query(q)

                # Handle abstention (ConfidentRAG mode)
                abstained = getattr(result, 'abstained', False)
                abstention_reason = getattr(result, 'abstention_reason', '')
                confidence = getattr(result, 'confidence_score', 1.0)

                # For abstained responses, use the abstention message as answer for eval
                if abstained:
                    answer = f"I cannot reliably answer this question. {abstention_reason}"
                else:
                    answer = result.answer

                samples.append(EvalSample(
                    question=q,
                    answer=answer,
                    contexts=[c.content for c in result.retrieved_chunks],
                    ground_truth="",  # Will be filled if available
                    abstained=abstained,
                    abstention_reason=abstention_reason,
                    confidence_score=confidence,
                ))
            except Exception as e:
                logger.warning(f"Error on question '{q}': {e}")
                samples.append(EvalSample(question=q, answer="", ground_truth=""))

            # Rate limiting delay between queries (skip after last one)
            if delay_seconds > 0 and i < len(questions) - 1:
                time.sleep(delay_seconds)

        return samples
    
    def quick_eval(
        self,
        questions: list[str],
        verbose: bool = True,
        delay_seconds: float = 0,
        run_name: Optional[str] = None,
    ) -> EvalResult:
        """
        Quick evaluation using LLM-as-Judge.

        No ground truth needed - judges based on:
        - Faithfulness to context
        - Relevancy to question
        - Context quality

        Args:
            questions: List of questions to evaluate
            verbose: Print progress
            delay_seconds: Delay between RAG queries to avoid rate limiting
            run_name: Optional name for MLflow run

        Returns:
            EvalResult with scores
        """
        if verbose:
            print("\n🔍 Running Quick Evaluation (LLM-as-Judge)")
            print("=" * 50)

        # Run RAG on questions
        samples = self.run_rag(questions, delay_seconds=delay_seconds)

        # Evaluate with LLM judge
        result = self._llm_judge.evaluate(samples, verbose=verbose)

        # Log to MLflow if enabled
        if self._mlflow_tracker:
            run_name = run_name or f"quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with self._mlflow_tracker.track_evaluation(run_name, tags={"eval_type": "quick"}):
                self._mlflow_tracker.log_evaluation_result(result)
                logger.info(f"Evaluation logged to MLflow run: {run_name}")

        if verbose:
            print(result)

        return result
    
    def full_eval(
        self,
        test_set: list[dict],
        verbose: bool = True,
        delay_seconds: float = 0,
        run_name: Optional[str] = None,
    ) -> EvalResult:
        """
        Full evaluation with ground truth.

        Uses RAGAS if OpenAI available, otherwise LLM-as-Judge + SemanticF1.

        Args:
            test_set: List of {"question": ..., "expected_answer": ...}
            verbose: Print progress
            delay_seconds: Delay between RAG queries to avoid rate limiting
            run_name: Optional name for MLflow run

        Returns:
            EvalResult with comprehensive scores
        """
        if verbose:
            print("\n🔍 Running Full Evaluation")
            print("=" * 50)

        # Run RAG and collect samples
        questions = [t["question"] for t in test_set]
        samples = self.run_rag(questions, delay_seconds=delay_seconds)

        # Add ground truth
        for sample, test in zip(samples, test_set):
            sample.ground_truth = test.get("expected_answer", "")

        # Calculate abstention metrics
        abstained_samples = [s for s in samples if s.abstained]
        answered_samples = [s for s in samples if not s.abstained]
        abstention_rate = len(abstained_samples) / len(samples) if samples else 0
        avg_confidence = (
            sum(s.confidence_score for s in answered_samples) / len(answered_samples)
            if answered_samples else 0
        )

        if verbose and abstained_samples:
            print(f"  Abstained on {len(abstained_samples)}/{len(samples)} questions ({abstention_rate:.1%})")
            print(f"  Average confidence on answered: {avg_confidence:.1%}")

        # For evaluation, only evaluate non-abstained samples
        # Abstention = refusing to hallucinate, which is good for faithfulness
        eval_samples = answered_samples if answered_samples else samples

        # Try RAGAS first
        ragas = self._get_ragas()
        if ragas:
            if verbose:
                print("Using RAGAS evaluation (OpenAI detected)")
            result = ragas.evaluate(eval_samples)
        else:
            # Fallback to LLM-as-Judge + SemanticF1
            if verbose:
                print("Using LLM-as-Judge + SemanticF1 evaluation")

            # LLM Judge for faithfulness/relevancy/context
            judge_result = self._llm_judge.evaluate(samples, verbose=verbose)

            # SemanticF1 for answer correctness
            correctness_scores = []
            for sample in samples:
                if sample.ground_truth:
                    score = self._dspy_eval.evaluate_single(
                        sample.answer, sample.ground_truth
                    )
                    correctness_scores.append(score)

            if correctness_scores:
                avg_correctness = sum(correctness_scores) / len(correctness_scores)
                judge_result.scores.answer_correctness = avg_correctness

                # Recalculate overall
                valid = [s for s in [
                    judge_result.scores.faithfulness,
                    judge_result.scores.answer_relevancy,
                    judge_result.scores.context_relevance,
                    judge_result.scores.answer_correctness,
                ] if s > 0]
                judge_result.scores.overall = sum(valid) / len(valid) if valid else 0

            result = judge_result

        # Add abstention metrics
        result.scores.abstention_rate = abstention_rate
        result.scores.answered_faithfulness = result.scores.faithfulness
        result.scores.avg_confidence = avg_confidence

        # Update num_samples to reflect actual evaluation
        result.num_samples = len(samples)

        # Log to MLflow if enabled
        if self._mlflow_tracker:
            run_name = run_name or f"full_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with self._mlflow_tracker.track_evaluation(run_name, tags={"eval_type": "full", "num_samples": str(len(test_set))}):
                self._mlflow_tracker.log_evaluation_result(result)
                logger.info(f"Evaluation logged to MLflow run: {run_name}")

        if verbose:
            print(result)

        return result
    
    def generate_report(
        self,
        result: EvalResult,
        output_path: str | Path,
    ):
        """Generate a detailed evaluation report."""
        result.save(output_path)
        
        # Also generate markdown summary
        md_path = Path(output_path).with_suffix('.md')
        
        md_content = f"""# RAG Pipeline Evaluation Report

Generated: {result.timestamp}

## Summary

| Metric | Score |
|--------|-------|
| **Overall** | {result.scores.overall:.1%} |
| Context Precision | {result.scores.context_precision:.1%} |
| Context Recall | {result.scores.context_recall:.1%} |
| Context Relevance | {result.scores.context_relevance:.1%} |
| Faithfulness | {result.scores.faithfulness:.1%} |
| Answer Relevancy | {result.scores.answer_relevancy:.1%} |
| Answer Correctness | {result.scores.answer_correctness:.1%} |

## Configuration

```json
{json.dumps(result.config, indent=2)}
```

## Samples Evaluated: {result.num_samples}

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
"""
        
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Report saved to {md_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Example Test Set
# ═══════════════════════════════════════════════════════════════════════════════

EXAMPLE_TEST_SET = [
    {
        "question": "What is RAG and how does it work?",
        "expected_answer": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by retrieving relevant documents from a knowledge base and using them as context for generation.",
    },
    {
        "question": "What are the benefits of using Docling for PDF processing?",
        "expected_answer": "Docling provides intelligent PDF parsing with layout analysis, table detection, and OCR support for scanned documents.",
    },
    {
        "question": "How does hybrid retrieval improve search results?",
        "expected_answer": "Hybrid retrieval combines BM25 keyword search with vector similarity search using Reciprocal Rank Fusion, getting the best of both exact matching and semantic understanding.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for RAG evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Quick eval
    quick_parser = subparsers.add_parser("quick", help="Quick LLM-as-Judge evaluation")
    quick_parser.add_argument("--questions", "-q", nargs="+",
                              help="Questions to evaluate")
    quick_parser.add_argument("--file", "-f", help="JSON file with questions")
    quick_parser.add_argument("--delay", "-d", type=float, default=0,
                              help="Delay between queries in seconds (default: 0)")
    quick_parser.add_argument("--provider", "-p", default="deepseek",
                              choices=["deepseek", "groq", "gemini"],
                              help="LLM provider (default: deepseek)")
    quick_parser.add_argument("--model", "-m",
                              help="LLM model name (e.g., deepseek-chat, deepseek-reasoner)")
    quick_parser.add_argument("--mlflow", action="store_true",
                              help="Enable MLflow tracking")
    quick_parser.add_argument("--run-name", help="MLflow run name")
    quick_parser.add_argument("--faithful", choices=["fast", "full", "confident", "confident-lite"],
                              help="Enable claim verification (fast=batch, full=individual, confident=with abstention)")
    quick_parser.add_argument("--evaluator-model", default="gpt-4o-mini",
                              help="Model for RAGAS evaluation (default: gpt-4o-mini)")
    quick_parser.add_argument("--reasoning-effort", default=None,
                              choices=["minimal", "low", "medium", "high"],
                              help="Reasoning effort for GPT-5 evaluator (optional)")

    # Full eval
    full_parser = subparsers.add_parser("full", help="Full evaluation with ground truth")
    full_parser.add_argument("--file", "-f", required=True,
                             help="JSON file with test set")
    full_parser.add_argument("--output", "-o", default="eval_results/results.json",
                             help="Output file for results (default: eval_results/results.json)")
    full_parser.add_argument("--delay", "-d", type=float, default=0,
                             help="Delay between queries in seconds (default: 0)")
    full_parser.add_argument("--provider", "-p", default="deepseek",
                             choices=["deepseek", "groq", "gemini"],
                             help="LLM provider (default: deepseek)")
    full_parser.add_argument("--model", "-m",
                             help="LLM model name (e.g., deepseek-chat, deepseek-reasoner)")
    full_parser.add_argument("--mlflow", action="store_true",
                             help="Enable MLflow tracking")
    full_parser.add_argument("--run-name", help="MLflow run name")
    full_parser.add_argument("--faithful", choices=["fast", "full", "confident", "confident-lite"],
                             help="Enable claim verification (fast=batch, full=individual, confident=with abstention)")
    full_parser.add_argument("--evaluator-model", default="gpt-4o-mini",
                             help="Model for RAGAS evaluation (default: gpt-4o-mini)")
    full_parser.add_argument("--reasoning-effort", default=None,
                             choices=["minimal", "low", "medium", "high"],
                             help="Reasoning effort for GPT-5 evaluator (optional)")
    
    # Demo
    subparsers.add_parser("demo", help="Run evaluation demo")
    
    args = parser.parse_args()
    
    if args.command == "quick":
        from rag_pipeline import RAGSystem

        questions = args.questions or [
            "What is the main topic of the documents?",
            "What are the key findings?",
        ]

        if args.file:
            with open(args.file) as f:
                questions = json.load(f)

        faithful_mode = getattr(args, 'faithful', None)
        provider = getattr(args, 'provider', 'deepseek')
        model = getattr(args, 'model', None)
        rag = RAGSystem(
            save_questions_to_faq=False,
            faithful_mode=faithful_mode,
            llm_provider=provider,
            llm_model=model,
        )
        print(f"Using provider: {provider}, model: {model or 'default'}")
        if faithful_mode:
            print(f"Using faithful mode: {faithful_mode}")
        evaluator_model = getattr(args, 'evaluator_model', 'gpt-4o-mini')
        reasoning_effort = getattr(args, 'reasoning_effort', None)
        print(f"Evaluator: {evaluator_model}")
        evaluator = PipelineEvaluator(
            rag,
            enable_mlflow=args.mlflow,
            evaluator_model=evaluator_model,
            reasoning_effort=reasoning_effort,
        )
        result = evaluator.quick_eval(
            questions,
            delay_seconds=args.delay,
            run_name=getattr(args, 'run_name', None)
        )
        print(result)

    elif args.command == "full":
        from rag_pipeline import RAGSystem

        with open(args.file) as f:
            test_set = json.load(f)

        faithful_mode = getattr(args, 'faithful', None)
        provider = getattr(args, 'provider', 'deepseek')
        model = getattr(args, 'model', None)
        rag = RAGSystem(
            save_questions_to_faq=False,
            k=5,
            faithful_mode=faithful_mode,
            llm_provider=provider,
            llm_model=model,
        )
        print(f"Using provider: {provider}, model: {model or 'default'}")
        if faithful_mode:
            print(f"Using faithful mode: {faithful_mode}")
        evaluator_model = getattr(args, 'evaluator_model', 'gpt-4o-mini')
        reasoning_effort = getattr(args, 'reasoning_effort', None)
        print(f"Evaluator: {evaluator_model}")
        evaluator = PipelineEvaluator(
            rag,
            enable_mlflow=args.mlflow,
            evaluator_model=evaluator_model,
            reasoning_effort=reasoning_effort,
        )
        result = evaluator.full_eval(
            test_set,
            delay_seconds=args.delay,
            run_name=getattr(args, 'run_name', None)
        )
        evaluator.generate_report(result, args.output)
    
    elif args.command == "demo":
        print("""
╔══════════════════════════════════════════════════════════════╗
║  RAG Evaluation Demo                                         ║
╚══════════════════════════════════════════════════════════════╝

Available Evaluation Approaches:

1. RAGAS (Industry Standard)
   - Requires OpenAI API key
   - Metrics: faithfulness, answer_relevancy, 
              context_precision, context_recall
   
2. DSPy SemanticF1
   - Uses your configured LLM
   - Compares predicted vs expected answers
   
3. LLM-as-Judge
   - Uses your configured LLM
   - Custom evaluation criteria
   - Provides reasoning for scores

Usage:

    # Quick evaluation (no ground truth needed)
    python evaluation.py quick -q "What is X?" "How does Y work?"
    
    # Full evaluation with ground truth
    python evaluation.py full -f test_set.json -o results.json
    
    # From Python:
    from evaluation import PipelineEvaluator
    from rag_pipeline import RAGSystem
    
    rag = RAGSystem(save_questions_to_faq=False)
    evaluator = PipelineEvaluator(rag)
    
    # Quick eval
    result = evaluator.quick_eval(["What is X?"])
    
    # Full eval
    result = evaluator.full_eval([
        {"question": "What is X?", "expected_answer": "X is..."}
    ])
    
    print(result)
""")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
