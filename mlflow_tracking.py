"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MLflow Tracking for RAG Pipeline                                            ║
║                                                                              ║
║  Provides experiment tracking, model versioning, and metrics logging:        ║
║  - Query-level tracking (latency, retrieval counts, answer quality)          ║
║  - Evaluation run tracking (all RAGAS/DSPy/LLM-Judge metrics)               ║
║  - Model configuration logging                                               ║
║  - Artifact storage (reports, embeddings, etc.)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import mlflow to allow graceful degradation
_mlflow = None


def _get_mlflow():
    """Lazy import of MLflow."""
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow
            _mlflow = mlflow
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow\n"
                "Or: uv add mlflow"
            )
    return _mlflow


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLflowConfig:
    """MLflow configuration settings."""
    tracking_uri: str = "mlruns"  # Local directory or remote URI
    experiment_name: str = "rag-pipeline"
    enable_system_metrics: bool = True
    log_artifacts: bool = True
    auto_log_models: bool = False  # DSPy models aren't standard ML models

    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "mlruns"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-pipeline"),
            enable_system_metrics=os.getenv("MLFLOW_SYSTEM_METRICS", "true").lower() == "true",
            log_artifacts=os.getenv("MLFLOW_LOG_ARTIFACTS", "true").lower() == "true",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Metrics Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryMetrics:
    """Metrics for a single RAG query."""
    question: str
    answer_length: int
    num_chunks_retrieved: int
    latency_ms: float
    has_reasoning: bool
    has_sources: bool
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class RetrievalMetrics:
    """Retrieval-specific metrics."""
    num_results: int
    avg_score: float
    sources: list[str]
    bm25_weight: float = 0.5
    vector_weight: float = 0.5


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics."""
    # Retrieval
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevance: float = 0.0

    # Generation
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    answer_correctness: float = 0.0

    # Combined
    overall: float = 0.0

    # Metadata
    num_samples: int = 0
    evaluator_type: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# MLflow Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class MLflowTracker:
    """
    MLflow tracking for RAG pipelines.

    Usage:
        tracker = MLflowTracker()

        # Track a RAG query
        with tracker.track_query("What is RAG?") as run:
            response = rag.query("What is RAG?")
            tracker.log_query_result(response)

        # Track an evaluation run
        with tracker.track_evaluation("full_eval_v1") as run:
            result = evaluator.full_eval(test_set)
            tracker.log_evaluation_result(result)

        # Log model configuration
        tracker.log_model_config({
            "llm_provider": "groq",
            "llm_model": "kimi-k2-instruct",
            "retrieval_k": 5,
            "hybrid_retrieval": True,
        })
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize MLflow tracker.

        Args:
            config: MLflow configuration (uses env vars if not provided)
        """
        self.config = config or MLflowConfig.from_env()
        self._mlflow = None
        self._experiment_id = None
        self._current_run = None
        self._query_start_time = None

    def _ensure_initialized(self):
        """Ensure MLflow is initialized."""
        if self._mlflow is None:
            self._mlflow = _get_mlflow()
            self._mlflow.set_tracking_uri(self.config.tracking_uri)

            # Create or get experiment
            experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self._experiment_id = self._mlflow.create_experiment(
                    self.config.experiment_name,
                    tags={"project": "dspy-supabase-rag"}
                )
            else:
                self._experiment_id = experiment.experiment_id

            self._mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLflow initialized: {self.config.tracking_uri}")

    @contextmanager
    def track_query(self, question: str, run_name: Optional[str] = None):
        """
        Context manager for tracking a RAG query.

        Args:
            question: The question being asked
            run_name: Optional name for the run

        Yields:
            MLflow run object
        """
        self._ensure_initialized()

        run_name = run_name or f"query_{datetime.now().strftime('%H%M%S')}"
        self._query_start_time = time.time()

        with self._mlflow.start_run(run_name=run_name) as run:
            self._current_run = run

            # Log query parameters
            self._mlflow.log_param("query_type", "rag_query")
            self._mlflow.log_param("question", question[:250])  # Truncate long questions
            self._mlflow.log_param("question_length", len(question))

            try:
                yield run
            finally:
                self._current_run = None
                self._query_start_time = None

    @contextmanager
    def track_evaluation(self, run_name: str, tags: Optional[dict] = None):
        """
        Context manager for tracking an evaluation run.

        Args:
            run_name: Name for the evaluation run
            tags: Optional tags for the run

        Yields:
            MLflow run object
        """
        self._ensure_initialized()

        with self._mlflow.start_run(run_name=run_name) as run:
            self._current_run = run

            # Set default tags
            self._mlflow.set_tag("run_type", "evaluation")
            if tags:
                for key, value in tags.items():
                    self._mlflow.set_tag(key, str(value))

            try:
                yield run
            finally:
                self._current_run = None

    @contextmanager
    def track_ingestion(self, source: str, run_name: Optional[str] = None):
        """
        Context manager for tracking document ingestion.

        Args:
            source: Document source being ingested
            run_name: Optional name for the run

        Yields:
            MLflow run object
        """
        self._ensure_initialized()

        run_name = run_name or f"ingest_{datetime.now().strftime('%H%M%S')}"
        start_time = time.time()

        with self._mlflow.start_run(run_name=run_name) as run:
            self._current_run = run

            self._mlflow.log_param("run_type", "ingestion")
            self._mlflow.log_param("source", str(source)[:250])

            try:
                yield run
            finally:
                # Log duration
                duration_ms = (time.time() - start_time) * 1000
                self._mlflow.log_metric("ingestion_duration_ms", duration_ms)
                self._current_run = None

    def log_query_result(self, response, retrieval_scores: Optional[list[float]] = None):
        """
        Log metrics from a RAG query response.

        Args:
            response: RAGResponse object from rag_pipeline.py
            retrieval_scores: Optional list of retrieval scores
        """
        if self._current_run is None:
            logger.warning("No active MLflow run. Use track_query() context manager.")
            return

        # Calculate latency
        latency_ms = 0
        if self._query_start_time:
            latency_ms = (time.time() - self._query_start_time) * 1000

        # Log metrics
        self._mlflow.log_metric("latency_ms", latency_ms)
        self._mlflow.log_metric("answer_length", len(response.answer))
        self._mlflow.log_metric("num_chunks_retrieved", len(response.retrieved_chunks))
        self._mlflow.log_metric("has_reasoning", 1 if response.reasoning else 0)
        self._mlflow.log_metric("has_sources", 1 if response.sources else 0)

        # Log retrieval scores if available
        if retrieval_scores:
            self._mlflow.log_metric("avg_retrieval_score", sum(retrieval_scores) / len(retrieval_scores))
            self._mlflow.log_metric("max_retrieval_score", max(retrieval_scores))
            self._mlflow.log_metric("min_retrieval_score", min(retrieval_scores))

        # Log chunk sources
        if response.retrieved_chunks:
            sources = list(set(c.source for c in response.retrieved_chunks))
            self._mlflow.log_param("sources", ", ".join(sources[:10]))  # Limit to 10

        # Log answer preview as artifact if enabled
        if self.config.log_artifacts:
            answer_data = {
                "question": response.question,
                "answer": response.answer,
                "reasoning": response.reasoning,
                "sources": response.sources,
                "num_chunks": len(response.retrieved_chunks),
            }

            # Create temp file and log
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(answer_data, f, indent=2)
                temp_path = f.name

            self._mlflow.log_artifact(temp_path, "responses")
            os.unlink(temp_path)

    def log_evaluation_result(self, result):
        """
        Log metrics from an evaluation result.

        Args:
            result: EvalResult object from evaluation.py
        """
        if self._current_run is None:
            logger.warning("No active MLflow run. Use track_evaluation() context manager.")
            return

        scores = result.scores

        # Log all metrics
        metrics = {
            # Retrieval metrics
            "context_precision": scores.context_precision,
            "context_recall": scores.context_recall,
            "context_relevance": scores.context_relevance,

            # Generation metrics
            "faithfulness": scores.faithfulness,
            "answer_relevancy": scores.answer_relevancy,
            "answer_correctness": scores.answer_correctness,

            # Overall
            "overall_score": scores.overall,

            # Metadata
            "num_samples": result.num_samples,
        }

        for name, value in metrics.items():
            if value is not None:
                self._mlflow.log_metric(name, float(value))

        # Log configuration
        if result.config:
            for key, value in result.config.items():
                self._mlflow.log_param(f"eval_{key}", str(value))

        # Log full result as artifact
        if self.config.log_artifacts:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(result.to_dict(), f, indent=2)
                temp_path = f.name

            self._mlflow.log_artifact(temp_path, "evaluation_results")
            os.unlink(temp_path)

    def log_model_config(self, config: dict):
        """
        Log RAG model/pipeline configuration.

        Args:
            config: Dictionary of configuration parameters
        """
        self._ensure_initialized()

        # If not in a run, create one for config logging
        if self._current_run is None:
            with self._mlflow.start_run(run_name="model_config") as run:
                self._mlflow.set_tag("run_type", "config")
                for key, value in config.items():
                    self._mlflow.log_param(key, str(value))
        else:
            for key, value in config.items():
                self._mlflow.log_param(key, str(value))

    def log_ingestion_result(self, num_chunks: int, source: str, metadata: Optional[dict] = None):
        """
        Log metrics from document ingestion.

        Args:
            num_chunks: Number of chunks created
            source: Document source
            metadata: Optional metadata about the ingestion
        """
        if self._current_run is None:
            logger.warning("No active MLflow run. Use track_ingestion() context manager.")
            return

        self._mlflow.log_metric("num_chunks", num_chunks)
        self._mlflow.log_param("document_source", str(source)[:250])

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    self._mlflow.log_metric(f"meta_{key}", value)
                else:
                    self._mlflow.log_param(f"meta_{key}", str(value)[:250])

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file.

        Args:
            local_path: Path to the local file
            artifact_path: Optional subdirectory in artifact store
        """
        if self._current_run is None:
            logger.warning("No active MLflow run.")
            return

        self._mlflow.log_artifact(local_path, artifact_path)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if self._current_run is None:
            logger.warning("No active MLflow run.")
            return

        self._mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        if self._current_run is None:
            logger.warning("No active MLflow run.")
            return

        self._mlflow.log_metrics(metrics, step=step)

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if self._current_run is None:
            logger.warning("No active MLflow run.")
            return

        self._mlflow.set_tag(key, value)

    def get_tracking_uri(self) -> str:
        """Get the MLflow tracking URI."""
        self._ensure_initialized()
        return self._mlflow.get_tracking_uri()

    def get_experiment_url(self) -> str:
        """Get URL to view experiment in MLflow UI."""
        self._ensure_initialized()
        tracking_uri = self.get_tracking_uri()
        if tracking_uri.startswith("http"):
            return f"{tracking_uri}/#/experiments/{self._experiment_id}"
        return f"Run 'mlflow ui' and open http://localhost:5000/#/experiments/{self._experiment_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# Decorator for Easy Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def track_rag_query(tracker: Optional[MLflowTracker] = None):
    """
    Decorator to automatically track RAG queries.

    Usage:
        tracker = MLflowTracker()

        @track_rag_query(tracker)
        def query(self, question: str) -> RAGResponse:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, question: str, *args, **kwargs):
            if tracker is None:
                return func(self, question, *args, **kwargs)

            with tracker.track_query(question):
                response = func(self, question, *args, **kwargs)
                tracker.log_query_result(response)
                return response

        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# RAG System Integration Mixin
# ═══════════════════════════════════════════════════════════════════════════════

class MLflowRAGMixin:
    """
    Mixin to add MLflow tracking to RAGSystem.

    Usage:
        class TrackedRAGSystem(MLflowRAGMixin, RAGSystem):
            pass

        rag = TrackedRAGSystem(enable_mlflow=True)
        response = rag.query("What is RAG?")  # Automatically tracked
    """

    _mlflow_tracker: Optional[MLflowTracker] = None

    def init_mlflow(self, config: Optional[MLflowConfig] = None):
        """Initialize MLflow tracking."""
        self._mlflow_tracker = MLflowTracker(config)

        # Log initial model configuration
        model_config = {
            "llm_provider": getattr(self, 'llm_provider', 'unknown'),
            "hybrid_retrieval": hasattr(self, 'retriever') and hasattr(self.retriever, '_bm25'),
            "k": getattr(self.rag, 'k', 5) if hasattr(self, 'rag') else 5,
            "use_reasoning": getattr(self.rag, 'use_reasoning', True) if hasattr(self, 'rag') else True,
            "save_questions_to_faq": getattr(self, 'save_questions_to_faq', True),
        }
        self._mlflow_tracker.log_model_config(model_config)

        logger.info(f"MLflow tracking enabled: {self._mlflow_tracker.get_experiment_url()}")

    def query_with_tracking(self, question: str):
        """Query with automatic MLflow tracking."""
        if self._mlflow_tracker is None:
            return self.query(question)

        with self._mlflow_tracker.track_query(question):
            response = self.query(question)
            self._mlflow_tracker.log_query_result(response)
            return response

    def ingest_with_tracking(self, source, **kwargs):
        """Ingest with automatic MLflow tracking."""
        if self._mlflow_tracker is None:
            return self.ingest(source, **kwargs)

        with self._mlflow_tracker.track_ingestion(str(source)):
            count = self.ingest(source, **kwargs)
            self._mlflow_tracker.log_ingestion_result(count, str(source), kwargs.get('metadata'))
            return count


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_default_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    """Get or create the default MLflow tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = MLflowTracker()
    return _default_tracker


def start_run(run_name: str, **kwargs):
    """Start a new MLflow run using the default tracker."""
    tracker = get_tracker()
    tracker._ensure_initialized()
    return tracker._mlflow.start_run(run_name=run_name, **kwargs)


def log_rag_metrics(
    question: str,
    answer: str,
    latency_ms: float,
    num_chunks: int,
    **extra_metrics
):
    """
    Convenience function to log RAG metrics without full response object.

    Args:
        question: The question asked
        answer: The generated answer
        latency_ms: Query latency in milliseconds
        num_chunks: Number of chunks retrieved
        **extra_metrics: Additional metrics to log
    """
    tracker = get_tracker()
    tracker._ensure_initialized()

    with tracker._mlflow.start_run(run_name=f"query_{datetime.now().strftime('%H%M%S')}"):
        tracker._mlflow.log_param("question", question[:250])
        tracker._mlflow.log_metric("latency_ms", latency_ms)
        tracker._mlflow.log_metric("answer_length", len(answer))
        tracker._mlflow.log_metric("num_chunks", num_chunks)

        for name, value in extra_metrics.items():
            if isinstance(value, (int, float)):
                tracker._mlflow.log_metric(name, value)
            else:
                tracker._mlflow.log_param(name, str(value))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for MLflow tracking utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="MLflow tracking for RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start MLflow UI")
    ui_parser.add_argument("--port", "-p", type=int, default=5000, help="Port number")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host address")

    # Info command
    subparsers.add_parser("info", help="Show MLflow configuration")

    # List runs command
    list_parser = subparsers.add_parser("list", help="List recent runs")
    list_parser.add_argument("--limit", "-n", type=int, default=10, help="Number of runs")

    args = parser.parse_args()

    if args.command == "ui":
        import subprocess
        config = MLflowConfig.from_env()
        print(f"Starting MLflow UI at http://{args.host}:{args.port}")
        print(f"Tracking URI: {config.tracking_uri}")
        subprocess.run([
            "mlflow", "ui",
            "--backend-store-uri", config.tracking_uri,
            "--host", args.host,
            "--port", str(args.port),
        ])

    elif args.command == "info":
        config = MLflowConfig.from_env()
        print(f"""
MLflow Configuration
════════════════════
Tracking URI:     {config.tracking_uri}
Experiment Name:  {config.experiment_name}
System Metrics:   {config.enable_system_metrics}
Log Artifacts:    {config.log_artifacts}

Environment Variables:
  MLFLOW_TRACKING_URI     - Set tracking server URI
  MLFLOW_EXPERIMENT_NAME  - Set experiment name
  MLFLOW_SYSTEM_METRICS   - Enable/disable system metrics
  MLFLOW_LOG_ARTIFACTS    - Enable/disable artifact logging
""")

    elif args.command == "list":
        tracker = get_tracker()
        tracker._ensure_initialized()

        mlflow = tracker._mlflow
        experiment = mlflow.get_experiment_by_name(tracker.config.experiment_name)

        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=args.limit,
                order_by=["start_time DESC"],
            )

            print(f"\nRecent Runs (Experiment: {tracker.config.experiment_name})")
            print("=" * 80)

            if len(runs) == 0:
                print("No runs found.")
            else:
                for _, run in runs.iterrows():
                    run_name = run.get("tags.mlflow.runName", "unnamed")
                    status = run.get("status", "unknown")
                    start = run.get("start_time", "")
                    print(f"  {run_name:<30} {status:<10} {start}")
        else:
            print(f"Experiment '{tracker.config.experiment_name}' not found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
