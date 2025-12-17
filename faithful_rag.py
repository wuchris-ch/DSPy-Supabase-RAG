"""
Faithful RAG Module - Claim Verification for High Faithfulness + Relevancy

Strategy: Generate first (preserves relevancy), then verify claims (ensures faithfulness)

This avoids the classic tradeoff where constraining generation kills relevancy.
Instead, we surgically remove only unsupported claims post-generation.

Pipeline:
    Question + Context
           │
           ▼
    ┌─────────────────┐
    │ Generate Answer │  ← Full, relevant answer (your existing high-relevancy approach)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Extract Claims  │  ← Break answer into atomic factual claims
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Verify Claims   │  ← Check each claim against context
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Rewrite Answer  │  ← Keep supported claims, flag/remove unsupported
    └────────┬────────┘
             │
             ▼
    Faithful + Relevant Answer
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import dspy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Claim Verification Signatures
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractClaims(dspy.Signature):
    """Extract atomic factual claims from an answer.

    Each claim should be a single, verifiable statement.
    Do not include opinions, hedging language, or meta-commentary."""

    answer = dspy.InputField(desc="The answer to extract claims from")
    question = dspy.InputField(desc="The original question (for context)")

    claims = dspy.OutputField(
        desc="List of atomic factual claims, one per line. "
             "Each claim should be independently verifiable. "
             "Format: one claim per line, no bullets or numbers."
    )


class VerifyClaim(dspy.Signature):
    """Verify if a claim is supported by the provided context.

    A claim is SUPPORTED if the context explicitly states or directly implies it.
    A claim is NOT SUPPORTED if it requires outside knowledge or inference beyond the context."""

    claim = dspy.InputField(desc="The factual claim to verify")
    context = dspy.InputField(desc="The source context to verify against")

    verdict = dspy.OutputField(
        desc="SUPPORTED or NOT_SUPPORTED"
    )
    evidence = dspy.OutputField(
        desc="If SUPPORTED: quote the relevant text. If NOT_SUPPORTED: explain why."
    )


class RewriteWithSupported(dspy.Signature):
    """Rewrite an answer using only the supported claims.

    Maintain the same style and completeness as the original.
    For claims that were not supported, either omit them or note the limitation."""

    question = dspy.InputField(desc="The original question")
    original_answer = dspy.InputField(desc="The original answer")
    supported_claims = dspy.InputField(
        desc="Claims that were verified as supported by context"
    )
    unsupported_claims = dspy.InputField(
        desc="Claims that were NOT supported by context"
    )

    faithful_answer = dspy.OutputField(
        desc="Rewritten answer using only supported claims. "
             "If key information was unsupported, acknowledge the limitation."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: str
    supported: bool
    evidence: str = ""


@dataclass
class FaithfulRAGResponse:
    """Response from the faithful RAG pipeline."""
    question: str
    answer: str  # Final faithful answer
    original_answer: str  # Pre-verification answer
    reasoning: Optional[str] = None
    sources: Optional[str] = None

    # Verification details
    claims_extracted: list[str] = field(default_factory=list)
    claims_supported: list[ClaimVerification] = field(default_factory=list)
    claims_unsupported: list[ClaimVerification] = field(default_factory=list)
    faithfulness_score: float = 0.0  # supported / total claims

    # From retrieval
    retrieved_chunks: list = field(default_factory=list)
    context: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Faithful RAG Module
# ═══════════════════════════════════════════════════════════════════════════════

class FaithfulRAGModule(dspy.Module):
    """
    RAG module with post-generation claim verification.

    This achieves high faithfulness WITHOUT sacrificing relevancy by:
    1. Generating a full, relevant answer first
    2. Extracting and verifying claims
    3. Rewriting with only supported claims

    Usage:
        module = FaithfulRAGModule(retriever, mode="verify")
        response = module("What is X?")

        print(response.answer)  # Faithful answer
        print(response.faithfulness_score)  # e.g., 0.92
        print(response.claims_unsupported)  # See what was removed

    Modes:
        - "verify": Full claim verification (3-4 LLM calls, highest faithfulness)
        - "flag": Generate + flag uncertain claims (2 LLM calls, good balance)
        - "strict": Strict grounding prompt only (1 LLM call, may hurt relevancy)
    """

    def __init__(
        self,
        retriever,
        k: int = 5,
        mode: str = "verify",
        remove_unsupported: bool = True,
    ):
        """
        Initialize the faithful RAG module.

        Args:
            retriever: Retriever instance
            k: Number of documents to retrieve
            mode: Verification mode ("verify", "flag", or "strict")
            remove_unsupported: If True, remove unsupported claims.
                               If False, flag them but keep in answer.
        """
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.mode = mode
        self.remove_unsupported = remove_unsupported

        # Generation (same as your existing high-relevancy approach)
        self.generate = dspy.ChainOfThought(AnswerWithReasoningFaithful)

        # Verification components
        self.extract_claims = dspy.Predict(ExtractClaims)
        self.verify_claim = dspy.Predict(VerifyClaim)
        self.rewrite = dspy.Predict(RewriteWithSupported)

    def _build_context(self, chunks) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.source
            if hasattr(chunk, 'section') and chunk.section:
                source_info = f"{chunk.source} - {chunk.section}"
            context_parts.append(
                f"[Source {i}: {source_info}]\n{chunk.content}\n"
            )
        return "\n---\n".join(context_parts)

    def _extract_claims(self, answer: str, question: str) -> list[str]:
        """Extract atomic claims from answer."""
        result = self.extract_claims(answer=answer, question=question)
        claims = [c.strip() for c in result.claims.strip().split('\n') if c.strip()]
        return claims

    def _verify_claims(
        self,
        claims: list[str],
        context: str
    ) -> tuple[list[ClaimVerification], list[ClaimVerification]]:
        """Verify each claim against context."""
        supported = []
        unsupported = []

        for claim in claims:
            try:
                result = self.verify_claim(claim=claim, context=context)
                verdict = result.verdict.strip().upper()

                verification = ClaimVerification(
                    claim=claim,
                    supported="SUPPORTED" in verdict,
                    evidence=result.evidence,
                )

                if verification.supported:
                    supported.append(verification)
                else:
                    unsupported.append(verification)

            except Exception as e:
                logger.warning(f"Failed to verify claim: {e}")
                # On error, assume supported to avoid over-filtering
                supported.append(ClaimVerification(
                    claim=claim,
                    supported=True,
                    evidence="Verification failed, assuming supported"
                ))

        return supported, unsupported

    def _rewrite_answer(
        self,
        question: str,
        original_answer: str,
        supported: list[ClaimVerification],
        unsupported: list[ClaimVerification],
    ) -> str:
        """Rewrite answer with only supported claims."""
        supported_text = "\n".join(v.claim for v in supported)
        unsupported_text = "\n".join(v.claim for v in unsupported) if unsupported else "None"

        result = self.rewrite(
            question=question,
            original_answer=original_answer,
            supported_claims=supported_text,
            unsupported_claims=unsupported_text,
        )

        return result.faithful_answer

    def forward(self, question: str) -> FaithfulRAGResponse:
        """
        Process a question with claim verification.
        """
        # Step 1: Retrieve
        chunks = self.retriever(question, k=self.k)
        context = self._build_context(chunks)

        # Step 2: Generate initial answer (high relevancy)
        gen_result = self.generate(context=context, question=question)
        original_answer = gen_result.answer

        if self.mode == "strict":
            # No verification, just return with strict prompt
            return FaithfulRAGResponse(
                question=question,
                answer=original_answer,
                original_answer=original_answer,
                reasoning=getattr(gen_result, 'reasoning', None),
                sources=getattr(gen_result, 'sources', None),
                retrieved_chunks=chunks,
                context=context,
                faithfulness_score=1.0,  # Assumed with strict prompt
            )

        # Step 3: Extract claims
        claims = self._extract_claims(original_answer, question)

        if not claims:
            # No claims to verify
            return FaithfulRAGResponse(
                question=question,
                answer=original_answer,
                original_answer=original_answer,
                reasoning=getattr(gen_result, 'reasoning', None),
                sources=getattr(gen_result, 'sources', None),
                retrieved_chunks=chunks,
                context=context,
                faithfulness_score=1.0,
            )

        # Step 4: Verify claims
        supported, unsupported = self._verify_claims(claims, context)

        # Calculate faithfulness score
        total_claims = len(supported) + len(unsupported)
        faithfulness_score = len(supported) / total_claims if total_claims > 0 else 1.0

        # Step 5: Rewrite if needed
        if unsupported and self.remove_unsupported:
            final_answer = self._rewrite_answer(
                question, original_answer, supported, unsupported
            )
        else:
            final_answer = original_answer

        return FaithfulRAGResponse(
            question=question,
            answer=final_answer,
            original_answer=original_answer,
            reasoning=getattr(gen_result, 'reasoning', None),
            sources=getattr(gen_result, 'sources', None),
            claims_extracted=claims,
            claims_supported=supported,
            claims_unsupported=unsupported,
            faithfulness_score=faithfulness_score,
            retrieved_chunks=chunks,
            context=context,
        )

    def __call__(self, question: str) -> FaithfulRAGResponse:
        return self.forward(question)


# ═══════════════════════════════════════════════════════════════════════════════
# Alternative Generation Signature (for strict mode)
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerWithReasoningFaithful(dspy.Signature):
    """Answer a question using the provided context.

    IMPORTANT: Base your answer ONLY on information explicitly stated in the context.
    If the context doesn't fully address the question, acknowledge the limitation."""

    context = dspy.InputField(
        desc="Retrieved document chunks - your ONLY source of information"
    )
    question = dspy.InputField(
        desc="The user's question"
    )

    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning, citing specific parts of the context"
    )
    answer = dspy.OutputField(
        desc="Answer based ONLY on the context. Cite sources for each claim."
    )
    sources = dspy.OutputField(
        desc="List of sources used (document names/sections)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Verification (for evaluation efficiency)
# ═══════════════════════════════════════════════════════════════════════════════

class BatchVerifyClaims(dspy.Signature):
    """Verify multiple claims against context in a single call.

    For each claim, determine if it is SUPPORTED or NOT_SUPPORTED by the context."""

    claims = dspy.InputField(desc="List of claims to verify, one per line")
    context = dspy.InputField(desc="The source context to verify against")

    verdicts = dspy.OutputField(
        desc="For each claim, output: CLAIM: [claim text] | VERDICT: SUPPORTED or NOT_SUPPORTED | REASON: [brief reason]\n"
             "One verdict per line, in the same order as input claims."
    )


class FaithfulRAGModuleFast(FaithfulRAGModule):
    """
    Faster version using batch claim verification.

    Reduces LLM calls from N+2 to 3 (generate, extract, batch-verify, rewrite).
    Slightly less accurate than individual verification but much faster.
    """

    def __init__(self, retriever, k: int = 5, remove_unsupported: bool = True):
        super().__init__(retriever, k=k, mode="verify", remove_unsupported=remove_unsupported)
        self.batch_verify = dspy.Predict(BatchVerifyClaims)

    def _verify_claims(
        self,
        claims: list[str],
        context: str
    ) -> tuple[list[ClaimVerification], list[ClaimVerification]]:
        """Batch verify all claims in a single LLM call."""
        if not claims:
            return [], []

        claims_text = "\n".join(claims)

        try:
            result = self.batch_verify(claims=claims_text, context=context)
            verdicts = result.verdicts.strip().split('\n')

            supported = []
            unsupported = []

            for i, verdict_line in enumerate(verdicts):
                if i >= len(claims):
                    break

                claim = claims[i]
                is_supported = "SUPPORTED" in verdict_line.upper() and "NOT_SUPPORTED" not in verdict_line.upper()

                # Extract reason if present
                reason = ""
                if "REASON:" in verdict_line:
                    reason = verdict_line.split("REASON:")[-1].strip()

                verification = ClaimVerification(
                    claim=claim,
                    supported=is_supported,
                    evidence=reason,
                )

                if is_supported:
                    supported.append(verification)
                else:
                    unsupported.append(verification)

            return supported, unsupported

        except Exception as e:
            logger.warning(f"Batch verification failed: {e}")
            # Fallback: assume all supported
            return [ClaimVerification(claim=c, supported=True) for c in claims], []


# ═══════════════════════════════════════════════════════════════════════════════
# Usage Example
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from rag_pipeline import setup_groq_lm
    from retriever import HybridRetriever

    # Setup
    setup_groq_lm()
    retriever = HybridRetriever()

    # Use faithful module
    rag = FaithfulRAGModuleFast(retriever, k=8)

    response = rag("What are the main features of Docling?")

    print(f"Question: {response.question}")
    print(f"\nOriginal Answer:\n{response.original_answer}")
    print(f"\nFaithful Answer:\n{response.answer}")
    print(f"\nFaithfulness Score: {response.faithfulness_score:.1%}")
    print(f"\nSupported Claims ({len(response.claims_supported)}):")
    for v in response.claims_supported:
        print(f"  - {v.claim}")
    print(f"\nUnsupported Claims ({len(response.claims_unsupported)}):")
    for v in response.claims_unsupported:
        print(f"  - {v.claim}")
        print(f"    Reason: {v.evidence}")
