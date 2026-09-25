"""
Data models: every judgment the LLM makes (relevance, methodology takeaway, citation-worthiness, results implication) is wrapped in a GroundedClaim,
which pairs the claim with a quote the model says it pulled from the abstract. That quote gets checked against the actual abstract text in analyze.py. If it doesn't match, the paper gets flagged for human review.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    paper_id: str  # arXiv short id "2508.01234"
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    published: datetime
    url: str
    categories: List[str] = Field(default_factory=list)
    citation_count: Optional[int] = None
    venue: Optional[str] = None

class GroundedClaim(BaseModel):
    claim: str
    evidence_quote: str = Field(
        ...,
        description="Exact substring from the paper's abstract that supports the claim.",
    )
    verified: bool = Field(
        default=False,
        description="True only if evidence_quote was confirmed to actually appear in the abstract.",
    )

class PaperAnalysis(BaseModel):
    paper_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: int = Field(..., ge=0, le=10)
    relevance_reasoning: GroundedClaim
    methodology_takeaway: Optional[GroundedClaim] = None
    citation_worthy: bool
    citation_reasoning: GroundedClaim
    results_implication: Optional[GroundedClaim] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool
    review_reason: Optional[str] = None
    review_status: str = "pending"  # pending | approved | rejected, set by a human later
    model_version: str
    analyzed_at: datetime