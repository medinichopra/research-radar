"""
Calls Claude to assess a shortlisted paper against the research question.

1. Structured output is forced via tool_choice.

2. The model's own evidence_quote fields are NOT trusted. _verify_quote checks
   each one against the actual abstract text. If a quote doesn't check out, or
   confidence is low, the paper is flagged needs_human_review=True and excluded
   from the "citation-worthy" bucket until a human looks at it. 
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

import anthropic

from .schema import GroundedClaim, PaperAnalysis, PaperMetadata

ANALYSIS_TOOL = {
    "name": "record_analysis",
    "description": (
        "Record a structured analysis of how a paper relates to a research question. "
        "Every *_evidence_quote field must be an exact substring copied from the "
        "abstract text provided. If you can't find supporting text, lower confidence "
        "and score conservatively rather than inferring beyond what's stated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relevance_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "relevance_claim": {"type": "string"},
            "relevance_evidence_quote": {"type": "string"},
            "methodology_claim": {"type": "string"},
            "methodology_evidence_quote": {"type": "string"},
            "citation_worthy": {"type": "boolean"},
            "citation_claim": {"type": "string"},
            "citation_evidence_quote": {"type": "string"},
            "results_claim": {"type": "string"},
            "results_evidence_quote": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "relevance_score",
            "relevance_claim",
            "relevance_evidence_quote",
            "citation_worthy",
            "citation_claim",
            "citation_evidence_quote",
            "confidence",
        ],
    },
}

CONFIDENCE_REVIEW_THRESHOLD = 0.5


def _verify_quote(quote: Optional[str], abstract: str) -> bool:
    if not quote:
        return False
    return quote.strip().lower() in abstract.lower()


def _build_claim(
    data: dict, claim_key: str, quote_key: str, abstract: str, review_reasons: list
) -> Optional[GroundedClaim]:
    claim = data.get(claim_key)
    quote = data.get(quote_key)
    if not claim:
        return None
    verified = _verify_quote(quote, abstract)
    if not verified:
        review_reasons.append(f"{claim_key}: evidence quote not found verbatim in abstract")
    return GroundedClaim(claim=claim, evidence_quote=quote or "", verified=verified)


def analyze_paper(
    paper: PaperMetadata,
    research_question: str,
    similarity_score: float,
    model: str = "claude-sonnet-4-6",
) -> PaperAnalysis:
    client = anthropic.Anthropic()

    prompt = f"""Research question: {research_question}

Paper title: {paper.title}
Abstract: {paper.abstract}

Assess this paper against the research question: how relevant it is, what
methodology could be reused, whether it's worth citing, and what its results
imply for the feasibility/credibility of the research question above."""

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "record_analysis"},
        messages=[{"role": "user", "content": prompt}],
    )

    tool_use = next(b for b in resp.content if b.type == "tool_use")
    data = tool_use.input

    review_reasons: list = []

    relevance_claim = _build_claim(
        data, "relevance_claim", "relevance_evidence_quote", paper.abstract, review_reasons
    )
    methodology_claim = _build_claim(
        data, "methodology_claim", "methodology_evidence_quote", paper.abstract, review_reasons
    )
    citation_claim = _build_claim(
        data, "citation_claim", "citation_evidence_quote", paper.abstract, review_reasons
    )
    results_claim = _build_claim(
        data, "results_claim", "results_evidence_quote", paper.abstract, review_reasons
    )

    confidence = float(data["confidence"])
    if confidence < CONFIDENCE_REVIEW_THRESHOLD:
        review_reasons.append("low model confidence")

    return PaperAnalysis(
        paper_id=paper.paper_id,
        similarity_score=similarity_score,
        relevance_score=int(data["relevance_score"]),
        relevance_reasoning=relevance_claim,
        methodology_takeaway=methodology_claim,
        citation_worthy=bool(data["citation_worthy"]),
        citation_reasoning=citation_claim,
        results_implication=results_claim,
        confidence=confidence,
        needs_human_review=bool(review_reasons),
        review_reason="; ".join(review_reasons) if review_reasons else None,
        model_version=model,
        analyzed_at=datetime.now(timezone.utc),
    )