# src/lit_agent/retrieval/vector_searcher.py

from typing import List, Dict, Any

import numpy as np


class VectorSearcher:
    def __init__(
        self,
        corpus_store,
        research_weight: float = 0.7,
        abstract_weight: float = 0.3,
    ):
        self.corpus = corpus_store
        self.research_weight = research_weight
        self.abstract_weight = abstract_weight

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        query_embedding = query_embedding.astype("float32")

        research_scores = self.corpus.research_embeddings @ query_embedding
        abstract_scores = self.corpus.abstract_embeddings @ query_embedding

        final_scores = np.full(len(self.corpus), -1e9, dtype="float32")

        for i in range(len(self.corpus)):
            has_research = bool(self.corpus.has_research_content[i])
            has_abstract = bool(self.corpus.has_abstract_note[i])

            if has_research and has_abstract:
                final_scores[i] = (
                    self.research_weight * research_scores[i]
                    + self.abstract_weight * abstract_scores[i]
                )

            elif has_research and not has_abstract:
                final_scores[i] = research_scores[i]

            elif not has_research and has_abstract:
                final_scores[i] = abstract_scores[i]

            else:
                final_scores[i] = -1e9

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []

        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            record = self.corpus.get_record(idx)

            results.append({
                "rank": rank,
                "corpus_index": idx,
                "row_id": record.get("row_id"),

                "score": float(final_scores[idx]),
                "vector_score": float(final_scores[idx]),

                "research_score": (
                    float(research_scores[idx])
                    if self.corpus.has_research_content[idx]
                    else None
                ),
                "abstract_score": (
                    float(abstract_scores[idx])
                    if self.corpus.has_abstract_note[idx]
                    else None
                ),

                "has_research_content": bool(self.corpus.has_research_content[idx]),
                "has_abstract_note": bool(self.corpus.has_abstract_note[idx]),

                "record": record,
            })

        return results