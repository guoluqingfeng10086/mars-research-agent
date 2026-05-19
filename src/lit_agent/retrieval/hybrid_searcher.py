# src/lit_agent/retrieval/hybrid_searcher.py

from typing import List, Dict, Any

import numpy as np

from lit_agent.agents.query_analyzer import QueryPlan


class HybridSearcher:
    """
    Hybrid retrieval searcher.

    Retrieval pipeline:

    Stage 1. Parallel recall
        - Article-level vector recall
        - Article-level BM25 recall

    Stage 2. Candidate merge
        - Union vector candidates and BM25 candidates

    Stage 3. RRF fusion
        - Fuse vector_rank and bm25_rank
        - Do not directly add vector_score and bm25_score

    Stage 4. Phrase-aware reranking
        - Apply a light phrase_score boost after RRF
        - phrase_score should not affect first-stage BM25 recall
    """

    def __init__(
        self,
        corpus_store,
        vector_searcher,
        bm25_searcher,
        vector_recall_k: int = 200,
        bm25_recall_k: int = 200,
        rrf_k: int = 80,
        phrase_boost_weight: float = 0.05,
    ):
        self.corpus = corpus_store
        self.vector_searcher = vector_searcher
        self.bm25_searcher = bm25_searcher

        self.vector_recall_k = vector_recall_k
        self.bm25_recall_k = bm25_recall_k
        self.rrf_k = rrf_k
        self.phrase_boost_weight = phrase_boost_weight

    def search(
        self,
        query_plan: QueryPlan,
        query_embedding: np.ndarray,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:

        # ============================================================
        # Stage 1: Parallel recall
        # ============================================================

        vector_results = self.vector_searcher.search(
            query_embedding=query_embedding,
            top_k=self.vector_recall_k,
        )

        bm25_results = self.bm25_searcher.search(
            query_plan=query_plan,
            top_k=self.bm25_recall_k,
        )

        # ============================================================
        # Stage 2: Convert recall results to lookup maps
        # ============================================================

        vector_rank_map = {
            item["corpus_index"]: item["rank"]
            for item in vector_results
        }

        bm25_rank_map = {
            item["corpus_index"]: item["rank"]
            for item in bm25_results
        }

        vector_score_map = {
            item["corpus_index"]: item.get("vector_score")
            for item in vector_results
        }

        bm25_score_map = {
            item["corpus_index"]: item.get("bm25_score")
            for item in bm25_results
        }

        raw_bm25_score_map = {
            item["corpus_index"]: item.get("raw_bm25_score")
            for item in bm25_results
        }

        bm25_phrase_score_map = {
            item["corpus_index"]: item.get("phrase_score", 0.0)
            for item in bm25_results
        }

        bm25_phrase_matches_map = {
            item["corpus_index"]: item.get("phrase_matches", [])
            for item in bm25_results
        }

        # ============================================================
        # Stage 3: Candidate merge
        # ============================================================

        candidate_indices = sorted(
            set(vector_rank_map.keys()) | set(bm25_rank_map.keys())
        )

        results = []

        # ============================================================
        # Stage 4: RRF fusion + phrase-aware rerank
        # ============================================================

        for idx in candidate_indices:
            record = self.corpus.get_record(idx)

            vector_rank = vector_rank_map.get(idx)
            bm25_rank = bm25_rank_map.get(idx)

            rrf_score = self._compute_rrf_score(
                vector_rank=vector_rank,
                bm25_rank=bm25_rank,
            )

            phrase_score, phrase_matches = self._get_phrase_match_info(
                idx=idx,
                record=record,
                query_plan=query_plan,
                bm25_phrase_score_map=bm25_phrase_score_map,
                bm25_phrase_matches_map=bm25_phrase_matches_map,
            )

            phrase_boost = self._compute_phrase_boost(phrase_score)
            final_score = rrf_score * phrase_boost

            results.append({
                "corpus_index": idx,
                "row_id": record.get("row_id"),

                # Final ranking score
                "score": float(final_score),

                # RRF fusion information
                "rrf_score": float(rrf_score),
                "vector_rank": vector_rank,
                "bm25_rank": bm25_rank,

                # Phrase rerank information
                "phrase_score": float(phrase_score),
                "phrase_boost": float(phrase_boost),
                "phrase_matches": phrase_matches,

                # Raw retrieval scores for inspection only
                "vector_score": vector_score_map.get(idx),
                "bm25_score": bm25_score_map.get(idx),
                "raw_bm25_score": raw_bm25_score_map.get(idx),

                "record": record,
            })

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results

    def _compute_rrf_score(
        self,
        vector_rank,
        bm25_rank,
    ) -> float:
        """
        Compute Reciprocal Rank Fusion score.

        Only ranks are used here.
        vector_score and bm25_score are not directly added.
        """

        rrf_score = 0.0

        if vector_rank is not None:
            rrf_score += 1.0 / (self.rrf_k + vector_rank)

        if bm25_rank is not None:
            rrf_score += 1.0 / (self.rrf_k + bm25_rank)

        return rrf_score

    def _get_phrase_match_info(
        self,
        idx: int,
        record: Dict[str, Any],
        query_plan: QueryPlan,
        bm25_phrase_score_map: Dict[int, float],
        bm25_phrase_matches_map: Dict[int, List[Dict[str, Any]]],
    ):
        """
        Get phrase_score and phrase_matches.

        If the document is already in BM25 results, reuse the phrase score
        computed by BM25Searcher.

        If the document is only recalled by vector search, compute phrase
        matches here for reranking and explanation.
        """

        if idx in bm25_phrase_score_map:
            phrase_score = bm25_phrase_score_map[idx]
            phrase_matches = bm25_phrase_matches_map.get(idx, [])
        else:
            phrase_score, phrase_matches = self.bm25_searcher.score_query_phrases(
                record=record,
                query_phrases=query_plan.query_phrases,
            )

        return phrase_score, phrase_matches

    def _compute_phrase_boost(self, phrase_score: float) -> float:
        """
        Compute light phrase-aware boost after RRF.

        If phrase_boost_weight = 0.0, phrase matching is only used for display.
        """

        return 1.0 + self.phrase_boost_weight * phrase_score