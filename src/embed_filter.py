"""
Semantic subfield filter: embeds each new paper's title+abstract, embeds the
research question, and keeps only papers above a cosine similarity threshold.

This runs before the LLM step on purpose. Embeddings are cheap and catch
"same subfield" candidates that keyword search misses (different vocabulary,
same idea); the LLM step is expensive and reserved for papers that already
cleared this bar.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from .schema import PaperMetadata

_model_cache = {}

def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def filter_by_similarity(
    papers: List[PaperMetadata],
    research_question: str,
    model_name: str,
    threshold: float,
) -> List[Tuple[PaperMetadata, float]]:
    if not papers:
        return []

    model = get_model(model_name)
    q_emb = model.encode([research_question])[0]
    texts = [f"{p.title}. {p.abstract}" for p in papers]
    embs = model.encode(texts)

    q_norm = np.linalg.norm(q_emb) + 1e-9
    sims = (embs @ q_emb) / (np.linalg.norm(embs, axis=1) * q_norm + 1e-9)

    scored = list(zip(papers, (float(s) for s in sims)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(p, s) for p, s in scored if s >= threshold]