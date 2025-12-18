"""
Confidence-Gated RAG Module for High-Stakes Domains (Medical/Legal)

Key innovations:
1. Multi-signal confidence scoring
2. Abstention when confidence is below threshold
3. Source agreement detection (flags contradictions)
4. Context coverage estimation

Target: Push faithfulness from low 90s to high 90s by refusing to answer
rather than hallucinating.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import dspy

from faithful_rag import (
    FaithfulRAGModule,
    FaithfulRAGModuleFast,
    FaithfulRAGResponse,
    ClaimVerification,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Confidence Signals
# ==============================================================================

class AssessContextCoverage(dspy.Signature):
    """Assess how well the retrieved context covers the question.

    Consider: Does the context contain information directly relevant to answering
    the question? Are there gaps in the information needed?"""

    question = dspy.InputField(desc="The user's question")
    context = dspy.InputField(desc="The retrieved context passages")

    coverage_assessment = dspy.OutputField(
        desc="Brief assessment of how well the context covers the question"
    )
    coverage_score = dspy.OutputField(
        desc="Score from 0-10 where 10 means context fully covers the question, "
             "0 means context is completely irrelevant. Just the number."
    )


class DetectContradictions(dspy.Signature):
    """Check if the retrieved sources contain contradictory information.

    Look for conflicting facts, numbers, dates, or claims across sources."""

    context = dspy.InputField(desc="The retrieved context passages from multiple sources")
    question = dspy.InputField(desc="The question being asked (for relevance)")

    has_contradictions = dspy.OutputField(
        desc="YES if sources contain contradictory information relevant to the question, NO otherwise"
    )
    contradiction_details = dspy.OutputField(
        desc="If YES, describe the contradiction. If NO, say 'No contradictions found.'"
    )


class ClassifyQueryScope(dspy.Signature):
    """Determine if the question can be answered from the given context.

    Consider: Is this question within the domain of the documents?
    Does answering require information beyond what's in the context?"""

    question = dspy.InputField(desc="The user's question")
    context = dspy.InputField(desc="The retrieved context passages")

    in_scope = dspy.OutputField(
        desc="YES if the question can reasonably be answered from the context, "
             "NO if it requires outside knowledge or is out of domain"
    )
    scope_reasoning = dspy.OutputField(
        desc="Brief explanation of the scope assessment"
    )


# ==============================================================================
# Response Classes
# ==============================================================================

@dataclass
class ConfidenceSignals:
    """All signals contributing to confidence score."""
    faithfulness_score: float = 0.0  # From claim verification
    context_coverage: float = 0.0    # How well context covers question
    source_agreement: float = 1.0    # 1.0 if no contradictions, lower if conflicts
    in_scope: bool = True            # Whether question is in domain
    num_claims_verified: int = 0
    num_claims_supported: int = 0
    has_contradictions: bool = False
    contradiction_details: str = ""
    coverage_assessment: str = ""
    scope_reasoning: str = ""


@dataclass
class ConfidentRAGResponse:
    """Response from confidence-gated RAG."""
    question: str
    answer: str
    abstained: bool = False          # True if system refused to answer
    abstention_reason: str = ""      # Why it abstained

    # Confidence
    confidence_score: float = 0.0    # Aggregated confidence (0-1)
    confidence_signals: ConfidenceSignals = field(default_factory=ConfidenceSignals)

    # From FaithfulRAG
    original_answer: str = ""
    reasoning: Optional[str] = None
    sources: Optional[str] = None
    claims_extracted: list[str] = field(default_factory=list)
    claims_supported: list[ClaimVerification] = field(default_factory=list)
    claims_unsupported: list[ClaimVerification] = field(default_factory=list)
    faithfulness_score: float = 0.0

    # Retrieval
    retrieved_chunks: list = field(default_factory=list)
    context: str = ""

    # For evaluation compatibility
    @property
    def answer_for_eval(self) -> str:
        """Return answer or abstention message for evaluation."""
        if self.abstained:
            return f"I cannot reliably answer this question. {self.abstention_reason}"
        return self.answer


# ==============================================================================
# Confidence-Gated RAG Module
# ==============================================================================

class ConfidentRAGModule(dspy.Module):
    """
    RAG module with confidence scoring and abstention.

    For medical/legal domains where wrong answers are worse than no answer.

    Confidence is computed from:
    1. Faithfulness score (from claim verification)
    2. Context coverage (does context address the question?)
    3. Source agreement (are sources consistent?)
    4. Scope check (is question in domain?)

    If confidence < threshold, the system abstains with explanation.

    Usage:
        module = ConfidentRAGModule(retriever, confidence_threshold=0.85)
        response = module("What is the drug interaction?")

        if response.abstained:
            print(f"System declined: {response.abstention_reason}")
        else:
            print(f"Answer ({response.confidence_score:.0%} confident): {response.answer}")
    """

    def __init__(
        self,
        retriever,
        k: int = 10,  # Higher k for better recall
        confidence_threshold: float = 0.75,  # Abstain below this
        use_fast_verification: bool = True,  # Use batch verification
        check_contradictions: bool = True,
        check_scope: bool = True,
        check_coverage: bool = True,
    ):
        """
        Initialize confidence-gated RAG.

        Args:
            retriever: Retriever instance
            k: Number of documents to retrieve (higher for better recall)
            confidence_threshold: Minimum confidence to provide answer (0-1)
            use_fast_verification: Use batch claim verification (faster)
            check_contradictions: Enable contradiction detection
            check_scope: Enable scope classification
            check_coverage: Enable context coverage assessment
        """
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.check_contradictions = check_contradictions
        self.check_scope = check_scope
        self.check_coverage = check_coverage

        # Core faithful RAG
        if use_fast_verification:
            self.faithful_rag = FaithfulRAGModuleFast(retriever, k=k)
        else:
            self.faithful_rag = FaithfulRAGModule(retriever, k=k, mode="verify")

        # Confidence assessment modules
        self.assess_coverage = dspy.Predict(AssessContextCoverage)
        self.detect_contradictions = dspy.Predict(DetectContradictions)
        self.classify_scope = dspy.Predict(ClassifyQueryScope)

    def _compute_confidence(
        self,
        faithful_response: FaithfulRAGResponse,
        signals: ConfidenceSignals,
    ) -> float:
        """
        Compute aggregate confidence score from multiple signals.

        Weights:
        - Faithfulness: 40% (most important for medical/legal)
        - Context coverage: 30%
        - Source agreement: 20%
        - Scope: 10% (binary, acts as multiplier)
        """
        if not signals.in_scope:
            return 0.0  # Out of scope = zero confidence

        weights = {
            'faithfulness': 0.40,
            'coverage': 0.30,
            'agreement': 0.20,
            'base': 0.10,  # Base score if we got this far
        }

        score = (
            weights['faithfulness'] * signals.faithfulness_score +
            weights['coverage'] * signals.context_coverage +
            weights['agreement'] * signals.source_agreement +
            weights['base'] * 1.0
        )

        # Penalty for contradictions even if source_agreement is high
        if signals.has_contradictions:
            score *= 0.85

        # Bonus if all claims were supported
        if signals.num_claims_verified > 0 and signals.num_claims_supported == signals.num_claims_verified:
            score = min(1.0, score * 1.05)

        return min(1.0, max(0.0, score))

    def _assess_confidence_signals(
        self,
        question: str,
        context: str,
        faithful_response: FaithfulRAGResponse,
    ) -> ConfidenceSignals:
        """Gather all confidence signals."""
        signals = ConfidenceSignals()

        # 1. Faithfulness from claim verification
        signals.faithfulness_score = faithful_response.faithfulness_score
        signals.num_claims_verified = len(faithful_response.claims_supported) + len(faithful_response.claims_unsupported)
        signals.num_claims_supported = len(faithful_response.claims_supported)

        # 2. Context coverage
        if self.check_coverage:
            try:
                coverage_result = self.assess_coverage(
                    question=question,
                    context=context,
                )
                signals.coverage_assessment = coverage_result.coverage_assessment
                # Parse score (handle various formats)
                score_str = coverage_result.coverage_score.strip()
                try:
                    score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
                    signals.context_coverage = min(10, max(0, score)) / 10.0
                except ValueError:
                    signals.context_coverage = 0.5  # Default if parsing fails
            except Exception as e:
                logger.warning(f"Coverage assessment failed: {e}")
                signals.context_coverage = 0.5
        else:
            signals.context_coverage = 0.7  # Default assumption

        # 3. Source agreement (contradiction detection)
        if self.check_contradictions:
            try:
                contradiction_result = self.detect_contradictions(
                    context=context,
                    question=question,
                )
                has_contradictions = "YES" in contradiction_result.has_contradictions.upper()
                signals.has_contradictions = has_contradictions
                signals.contradiction_details = contradiction_result.contradiction_details
                signals.source_agreement = 0.5 if has_contradictions else 1.0
            except Exception as e:
                logger.warning(f"Contradiction detection failed: {e}")
                signals.source_agreement = 0.8  # Assume mostly agreeing
        else:
            signals.source_agreement = 1.0

        # 4. Scope classification
        if self.check_scope:
            try:
                scope_result = self.classify_scope(
                    question=question,
                    context=context,
                )
                signals.in_scope = "YES" in scope_result.in_scope.upper()
                signals.scope_reasoning = scope_result.scope_reasoning
            except Exception as e:
                logger.warning(f"Scope classification failed: {e}")
                signals.in_scope = True  # Default to in-scope
        else:
            signals.in_scope = True

        return signals

    def _determine_abstention_reason(self, signals: ConfidenceSignals) -> str:
        """Generate human-readable abstention reason."""
        reasons = []

        if not signals.in_scope:
            reasons.append(f"Question appears outside the scope of available documents. {signals.scope_reasoning}")

        if signals.context_coverage < 0.4:
            reasons.append(f"Retrieved documents don't adequately cover this topic. {signals.coverage_assessment}")

        if signals.has_contradictions:
            reasons.append(f"Sources contain conflicting information. {signals.contradiction_details}")

        if signals.faithfulness_score < 0.6:
            reasons.append(
                f"Could not verify enough claims from sources "
                f"({signals.num_claims_supported}/{signals.num_claims_verified} supported)."
            )

        if not reasons:
            reasons.append("Confidence in answer quality is below acceptable threshold.")

        return " ".join(reasons)

    def forward(self, question: str) -> ConfidentRAGResponse:
        """
        Process question with confidence gating.
        """
        # Step 1: Run faithful RAG
        faithful_response = self.faithful_rag(question)

        # Step 2: Assess confidence signals
        signals = self._assess_confidence_signals(
            question=question,
            context=faithful_response.context,
            faithful_response=faithful_response,
        )

        # Step 3: Compute aggregate confidence
        confidence = self._compute_confidence(faithful_response, signals)

        # Step 4: Decide whether to abstain
        should_abstain = confidence < self.confidence_threshold

        if should_abstain:
            abstention_reason = self._determine_abstention_reason(signals)
            return ConfidentRAGResponse(
                question=question,
                answer="",
                abstained=True,
                abstention_reason=abstention_reason,
                confidence_score=confidence,
                confidence_signals=signals,
                original_answer=faithful_response.original_answer,
                reasoning=faithful_response.reasoning,
                sources=faithful_response.sources,
                claims_extracted=faithful_response.claims_extracted,
                claims_supported=faithful_response.claims_supported,
                claims_unsupported=faithful_response.claims_unsupported,
                faithfulness_score=faithful_response.faithfulness_score,
                retrieved_chunks=faithful_response.retrieved_chunks,
                context=faithful_response.context,
            )

        # Return confident answer
        return ConfidentRAGResponse(
            question=question,
            answer=faithful_response.answer,
            abstained=False,
            abstention_reason="",
            confidence_score=confidence,
            confidence_signals=signals,
            original_answer=faithful_response.original_answer,
            reasoning=faithful_response.reasoning,
            sources=faithful_response.sources,
            claims_extracted=faithful_response.claims_extracted,
            claims_supported=faithful_response.claims_supported,
            claims_unsupported=faithful_response.claims_unsupported,
            faithfulness_score=faithful_response.faithfulness_score,
            retrieved_chunks=faithful_response.retrieved_chunks,
            context=faithful_response.context,
        )

    def __call__(self, question: str) -> ConfidentRAGResponse:
        return self.forward(question)


# ==============================================================================
# Simplified Confident RAG (fewer LLM calls)
# ==============================================================================

class ConfidentRAGModuleLite(dspy.Module):
    """
    Lighter version that skips some checks for speed.

    Only uses:
    - Claim verification (from FaithfulRAG)
    - Simple heuristics for context coverage

    Faster but slightly less accurate abstention decisions.
    """

    def __init__(
        self,
        retriever,
        k: int = 10,
        confidence_threshold: float = 0.60,  # Lower threshold to reduce abstention
    ):
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.faithful_rag = FaithfulRAGModuleFast(retriever, k=k)

    def _estimate_coverage(self, question: str, chunks: list) -> float:
        """Quick heuristic for context coverage without LLM call."""
        if not chunks:
            return 0.0

        # Check if question keywords appear in retrieved content
        question_words = set(question.lower().split())
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must', 'shall', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'about', 'against', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now'}

        content_words = question_words - stop_words

        if not content_words:
            return 0.5  # Can't assess without content words

        # Check how many content words appear in chunks
        all_content = " ".join(c.content.lower() for c in chunks)
        matches = sum(1 for w in content_words if w in all_content)

        keyword_coverage = matches / len(content_words)

        # Also consider similarity scores
        avg_similarity = sum(c.similarity_score for c in chunks if hasattr(c, 'similarity_score')) / len(chunks)

        # Combine heuristics
        return 0.6 * keyword_coverage + 0.4 * avg_similarity

    def forward(self, question: str) -> ConfidentRAGResponse:
        """Process question with lightweight confidence gating."""
        faithful_response = self.faithful_rag(question)

        # Quick coverage estimate
        coverage = self._estimate_coverage(question, faithful_response.retrieved_chunks)

        signals = ConfidenceSignals(
            faithfulness_score=faithful_response.faithfulness_score,
            context_coverage=coverage,
            source_agreement=1.0,  # Skip contradiction check
            in_scope=True,  # Skip scope check
            num_claims_verified=len(faithful_response.claims_supported) + len(faithful_response.claims_unsupported),
            num_claims_supported=len(faithful_response.claims_supported),
        )

        # Confidence: balanced weight - abstention happens when either is low
        confidence = 0.5 * signals.faithfulness_score + 0.5 * signals.context_coverage

        should_abstain = confidence < self.confidence_threshold

        if should_abstain:
            reason = f"Low confidence ({confidence:.0%}). "
            if signals.faithfulness_score < 0.7:
                reason += f"Only {signals.num_claims_supported}/{signals.num_claims_verified} claims verified. "
            if coverage < 0.5:
                reason += "Retrieved documents may not cover this topic well."

            return ConfidentRAGResponse(
                question=question,
                answer="",
                abstained=True,
                abstention_reason=reason,
                confidence_score=confidence,
                confidence_signals=signals,
                original_answer=faithful_response.original_answer,
                faithfulness_score=faithful_response.faithfulness_score,
                retrieved_chunks=faithful_response.retrieved_chunks,
                context=faithful_response.context,
            )

        return ConfidentRAGResponse(
            question=question,
            answer=faithful_response.answer,
            abstained=False,
            confidence_score=confidence,
            confidence_signals=signals,
            original_answer=faithful_response.original_answer,
            reasoning=faithful_response.reasoning,
            sources=faithful_response.sources,
            claims_extracted=faithful_response.claims_extracted,
            claims_supported=faithful_response.claims_supported,
            claims_unsupported=faithful_response.claims_unsupported,
            faithfulness_score=faithful_response.faithfulness_score,
            retrieved_chunks=faithful_response.retrieved_chunks,
            context=faithful_response.context,
        )

    def __call__(self, question: str) -> ConfidentRAGResponse:
        return self.forward(question)


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    from rag_pipeline import setup_deepseek_lm
    from retriever import HybridRetriever

    # Setup
    setup_deepseek_lm()
    retriever = HybridRetriever()

    # Use confident module
    rag = ConfidentRAGModule(
        retriever,
        k=8,
        confidence_threshold=0.75,
    )

    # Test query
    response = rag("What are the main features of Docling?")

    print(f"Question: {response.question}")
    print(f"\nAbstained: {response.abstained}")
    print(f"Confidence: {response.confidence_score:.1%}")

    if response.abstained:
        print(f"\nReason: {response.abstention_reason}")
    else:
        print(f"\nAnswer: {response.answer}")

    print(f"\n--- Confidence Signals ---")
    print(f"Faithfulness: {response.confidence_signals.faithfulness_score:.1%}")
    print(f"Coverage: {response.confidence_signals.context_coverage:.1%}")
    print(f"Source Agreement: {response.confidence_signals.source_agreement:.1%}")
    print(f"In Scope: {response.confidence_signals.in_scope}")
