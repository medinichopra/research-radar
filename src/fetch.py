"""
Pulls recent papers from arXiv for the configured categories.
fetch_recent_papers is the only function the rest of the pipeline depends on, so adding a second source
(Semantic Scholar, SSRN via a scraper, etc.) later would mean writing another function with the same return shape and merging the lists in pipeline.py.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import List

import arxiv
from .schema import PaperMetadata

log = logging.getLogger(__name__)


def fetch_recent_papers(
    categories: List[str], lookback_days: int, max_results: int = 100
) -> List[PaperMetadata]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    query = " OR ".join(f"cat:{c}" for c in categories)

    search = arxiv.Search(
        query=query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=max_results,
    )
    client = arxiv.Client()

    papers: List[PaperMetadata] = []
    for result in client.results(search):
        published = result.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            break
        papers.append(
            PaperMetadata(
                paper_id=result.get_short_id(),
                title=result.title.strip(),
                abstract=result.summary.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                published=published,
                url=result.entry_id,
                categories=list(result.categories),
            )
        )

    log.info(f"Fetched {len(papers)} papers within the last {lookback_days} days")
    return papers