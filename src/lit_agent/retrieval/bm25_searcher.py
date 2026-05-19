# src/lit_agent/retrieval/bm25_searcher.py

from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from lit_agent.agents.query_analyzer import QueryPlan
from lit_agent.utils.text_utils import (
    clean_text,
    normalize_phrase,
    normalize_text,
    phrase_ngram_tokenize,
)


class BM25Searcher:
    """
    BM25 searcher based on:
    - Title
    - research_region
    - research_content
    - data_instruments_samples

    Abstract Note is not used.
    Query uses complete phrases extracted by LLM.

    Important:
    - Stage 1 BM25 recall only uses raw BM25 scores.
    - phrase_score is only computed for later hybrid reranking / explanation.
    """

    def __init__(
        self,
        corpus_store,
        phrase_boost: float = 0.0,
        no_phrase_penalty: float = 0.0,
        max_ngram: int = 6,
    ):
        self.corpus = corpus_store

        # These two parameters are kept for compatibility,
        # but are no longer used in first-stage BM25 recall.
        self.phrase_boost = phrase_boost
        self.no_phrase_penalty = no_phrase_penalty

        self.max_ngram = max_ngram

        self.documents: List[str] = []
        self.tokenized_documents: List[List[str]] = []

        print("Building phrase-based BM25 index...")

        for record in self.corpus.records:
            doc_text = self._build_bm25_text(record)
            tokens = phrase_ngram_tokenize(doc_text, max_n=self.max_ngram)

            self.documents.append(doc_text)
            self.tokenized_documents.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_documents)

        print(f"BM25 index built. Documents: {len(self.documents)}")

    def _build_bm25_text(self, record: Dict[str, Any]) -> str:
        """
        Build article-level BM25 text.

        No field repetition here.
        Field-level importance is handled later by score_query_phrases().
        """

        title = clean_text(record.get("Title", ""))
        region = clean_text(record.get("research_region", ""))
        research_content = clean_text(record.get("research_content", ""))
        data_instruments = clean_text(record.get("data_instruments_samples", ""))

        return "\n".join([
            title,
            region,
            research_content,
            data_instruments,
        ])

    def search(self, query_plan: QueryPlan, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Stage 1 BM25 recall.

        This version uses only raw BM25 scores for recall ranking.
        phrase_score is computed after selecting BM25 candidates,
        and is returned for HybridSearcher reranking / explanation.

        It does NOT do:
            adjusted_score = raw_bm25_score * (1 + phrase_boost * phrase_score)

        Therefore, phrase_score does not affect first-stage BM25 recall.
        """

        query_tokens = [
            normalize_phrase(p)
            for p in query_plan.query_phrases
            if normalize_phrase(p)
        ]

        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)

        positive_indices = np.where(raw_scores > 0)[0]

        if len(positive_indices) == 0:
            return []

        top_indices = positive_indices[
            np.argsort(raw_scores[positive_indices])[::-1]
        ][:top_k]

        results = []

        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            record = self.corpus.get_record(idx)

            phrase_score, phrase_matches = self.score_query_phrases(
                record=record,
                query_phrases=query_plan.query_phrases,
            )

            results.append({
                "rank": rank,
                "corpus_index": idx,
                "row_id": record.get("row_id"),

                # BM25 score is now pure raw BM25 score.
                "bm25_score": float(raw_scores[idx]),
                "raw_bm25_score": float(raw_scores[idx]),

                "query_phrases": query_plan.query_phrases,

                # phrase_score is only for later reranking / explanation.
                "phrase_score": float(phrase_score),
                "phrase_matches": phrase_matches,

                "record": record,
            })

        return results

    def score_query_phrases(
        self,
        record: Dict[str, Any],
        query_phrases: List[str],
    ) -> Tuple[float, List[Dict[str, Any]]]:

        if not query_phrases:
            return 0.0, []

        title = normalize_text(record.get("Title", ""))
        region = normalize_text(record.get("research_region", ""))
        research_content = normalize_text(record.get("research_content", ""))
        data_instruments = normalize_text(record.get("data_instruments_samples", ""))

        matches = []
        total_score = 0.0

        for phrase in query_phrases:
            phrase_norm = normalize_phrase(phrase)

            if not phrase_norm:
                continue

            fields = []
            phrase_score = 0.0

            # Title / research_region:
            # If either one matches, add 0.40 only once.
            if phrase_norm in title:
                fields.append("Title")
                phrase_score += 0.3

            if phrase_norm in region:
                fields.append("research_region")
                if phrase_norm not in title:
                    phrase_score += 0.3

            # research_content: additional weak-to-medium evidence.
            if phrase_norm in research_content:
                fields.append("research_content")
                phrase_score += 0.2

            # data_instruments_samples: weak auxiliary evidence.
            if phrase_norm in data_instruments:
                fields.append("data_instruments_samples")
                phrase_score += 0.10

            if fields:
                matches.append({
                    "phrase": phrase,
                    "normalized_phrase": phrase_norm,
                    "fields": fields,
                    "score": min(phrase_score, 1.0),
                })

            total_score += min(phrase_score, 1.0)

        phrase_score = total_score / len(query_phrases)

        return min(phrase_score, 1.0), matches