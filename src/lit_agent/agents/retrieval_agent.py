# src/lit_agent/agents/retrieval_agent.py

from typing import Any, Dict, List, Optional


class RetrievalAgent:
    def __init__(self, embedder, searcher, query_analyzer):
        self.embedder = embedder
        self.searcher = searcher
        self.query_analyzer = query_analyzer
        self.last_query_plan = None

    @classmethod
    def from_config(cls):
        """
        Build the complete retrieval agent from project config.

        This moves the initialization logic originally placed in
        scripts/search_demo.py into the reusable RetrievalAgent.
        """
        from config import (
            CORPUS_PKL_PATH,
            MODEL_PATH,
            DEVICE,
            MAX_SEQ_LENGTH,
            RESEARCH_CONTENT_WEIGHT,
            ABSTRACT_NOTE_WEIGHT,
            VECTOR_RECALL_K,
            BM25_RECALL_K,
            RRF_K,
            PHRASE_BOOST_WEIGHT,
        )

        from lit_agent.data.corpus_store import CorpusStore
        from lit_agent.models.embedder import BGEEmbedder
        from lit_agent.agents.query_analyzer import QueryAnalyzer
        from lit_agent.retrieval.vector_searcher import VectorSearcher
        from lit_agent.retrieval.bm25_searcher import BM25Searcher
        from lit_agent.retrieval.hybrid_searcher import HybridSearcher

        print("Loading corpus...")
        corpus = CorpusStore(CORPUS_PKL_PATH).load()
        print(corpus.basic_info())

        print("Loading embedding model...")
        embedder = BGEEmbedder(
            model_path=MODEL_PATH,
            device=DEVICE,
            max_seq_length=MAX_SEQ_LENGTH,
            normalize_embeddings=True,
        )

        print("Building query analyzer...")
        query_analyzer = QueryAnalyzer()

        print("Building vector searcher...")
        vector_searcher = VectorSearcher(
            corpus_store=corpus,
            research_weight=RESEARCH_CONTENT_WEIGHT,
            abstract_weight=ABSTRACT_NOTE_WEIGHT,
        )

        print("Building BM25 searcher...")
        bm25_searcher = BM25Searcher(
            corpus_store=corpus,
        )

        print("Building hybrid searcher...")
        hybrid_searcher = HybridSearcher(
            corpus_store=corpus,
            vector_searcher=vector_searcher,
            bm25_searcher=bm25_searcher,
            vector_recall_k=VECTOR_RECALL_K,
            bm25_recall_k=BM25_RECALL_K,
            phrase_boost_weight=PHRASE_BOOST_WEIGHT,
            rrf_k=RRF_K,
        )

        return cls(
            embedder=embedder,
            searcher=hybrid_searcher,
            query_analyzer=query_analyzer,
        )

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        query_plan = self.query_analyzer.analyze(query)
        self.last_query_plan = query_plan

        query_embedding = self.embedder.encode_query(query_plan.semantic_query)

        results = self.searcher.search(
            query_plan=query_plan,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results

    def retrieve(self, query: str, top_k: int = 20, normalize: bool = True) -> List[Dict[str, Any]]:
        """
        Public retrieval interface for downstream agents.

        search():
            returns original hybrid search results.

        retrieve():
            returns standardized paper records, better for relevance/survey/mindmap agents.
        """
        results = self.search(query=query, top_k=top_k)

        if normalize:
            return [self.normalize_result_item(item) for item in results]

        return results

    def normalize_result_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert one raw search result into a stable paper schema.

        Downstream agents should use this standardized structure instead of
        directly depending on the original corpus field names.
        """
        record = item.get("record", {})

        paper_id = (
            record.get("paper_id")
            or record.get("id")
            or record.get("file_name")
            or record.get("Title")
        )

        return {
            "paper_id": paper_id,
            "rank": item.get("rank"),
            "score": item.get("score"),
            "rrf_score": item.get("rrf_score"),
            "phrase_score": item.get("phrase_score"),
            "phrase_boost": item.get("phrase_boost"),
            "phrase_matches": item.get("phrase_matches"),
            "vector_rank": item.get("vector_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "vector_score": item.get("vector_score"),
            "bm25_score": item.get("bm25_score"),
            "raw_bm25_score": item.get("raw_bm25_score"),

            "title": record.get("Title", ""),
            "year": record.get("Publication Year", ""),
            "journal": record.get("Publication Title", ""),
            "authors": record.get("Author", ""),
            "url": record.get("Url", ""),
            "file_name": record.get("file_name", ""),
            "pdf_path": record.get("pdf_path", "") or record.get("file_path", ""),
            "research_region": record.get("research_region", ""),
            "data_instruments_samples": record.get("data_instruments_samples", ""),
            "abstract_note": record.get("abstract_note", ""),
            "abstract": record.get("Abstract", "") or record.get("abstract", ""),
            "research_content": record.get("research_content", ""),

            # Keep original record for debugging or advanced use.
            #"record": record,
        }

    def format_query_plan(self) -> str:
        if self.last_query_plan is None:
            return ""

        qp = self.last_query_plan

        lines = []
        lines.append("=" * 100)
        lines.append("Query analysis")
        lines.append(f"Original query: {qp.original_query}")
        lines.append(f"Geological concepts: {qp.geological_concepts}")
        lines.append(f"Target regions: {qp.target_regions}")
        lines.append(f"BM25 query: {qp.bm25_query}")
        lines.append(f"Query phrases: {qp.query_phrases}")
        lines.append("=" * 100)

        return "\n".join(lines)

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format either raw search results or normalized retrieve results.
        """
        lines = []

        for item in results:
            # Compatible with both raw item and normalized item.
            record = item.get("record", item)

            title = item.get("title") or record.get("Title", "")
            year = item.get("year") or record.get("Publication Year", "")
            journal = item.get("journal") or record.get("Publication Title", "")
            authors = item.get("authors") or record.get("Author", "")
            region = item.get("research_region") or record.get("research_region", "")
            data_info = item.get("data_instruments_samples") or record.get("data_instruments_samples", "")
            file_name = item.get("file_name") or record.get("file_name", "")
            url = item.get("url") or record.get("Url", "")
            research_content = item.get("research_content") or record.get("research_content", "")

            lines.append("=" * 100)
            lines.append(f"Rank: {item.get('rank')}")
            score = item.get("score")
            lines.append(f"Final score: {score:.6f}" if isinstance(score, float) else f"Final score: {score}")

            lines.append(f"RRF score: {item.get('rrf_score')}")
            lines.append(f"Phrase score: {item.get('phrase_score')}")
            lines.append(f"Phrase boost: {item.get('phrase_boost')}")
            lines.append(f"Phrase matches: {item.get('phrase_matches')}")

            lines.append(f"Vector rank: {item.get('vector_rank')}")
            lines.append(f"BM25 rank: {item.get('bm25_rank')}")
            lines.append(f"Vector score: {item.get('vector_score')}")
            lines.append(f"BM25 score: {item.get('bm25_score')}")
            lines.append(f"Raw BM25 score: {item.get('raw_bm25_score')}")

            lines.append(f"Title: {title}")
            lines.append(f"Year: {year}")
            lines.append(f"Journal: {journal}")
            lines.append(f"Author: {authors}")
            lines.append(f"Region: {region}")
            lines.append(f"Data/Instruments/Samples: {data_info}")
            lines.append(f"File: {file_name}")
            lines.append(f"URL: {url}")
            lines.append("")
            lines.append("Research content:")
            lines.append(research_content)
            lines.append("")

        return "\n".join(lines)