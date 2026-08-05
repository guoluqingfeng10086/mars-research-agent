import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    DEFAULT_MAX_SOURCE_CHARS,
    SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS,
    SUPPLEMENT_ADS_API_TOKEN,
    SUPPLEMENT_FROM_YEAR,
    SUPPLEMENT_WEB_ENGINES,
    SUPPLEMENT_WEB_TIMEOUT,
)
from lit_agent.agents.paper_digest_agent import PaperDigestAgent
from lit_agent.agents.relevance_agent import RelevanceAgent
from lit_agent.agents.retrieval_agent import RetrievalAgent
from lit_agent.agents.supplement_agent import SupplementAgent
from lit_agent.retrieval.geo_context_retriever import query_and_summarize_geo_context
from lit_agent.retrieval.web_literature_searcher import WebLiteratureSearcher


class DiscoveryEvidenceAgent:
    """Acquire gap-directed evidence through existing project components."""

    def __init__(
        self,
        retrieval_agent: Optional[RetrievalAgent] = None,
        relevance_agent: Optional[RelevanceAgent] = None,
        supplement_agent: Optional[SupplementAgent] = None,
    ):
        self.retrieval_agent = retrieval_agent
        self.relevance_agent = relevance_agent
        self.supplement_agent = supplement_agent

    def _id(self, source_type: str, source_key: str, gap_id: str) -> str:
        raw = f"{source_type}|{source_key}|{gap_id}".lower().encode("utf-8")
        return "D" + hashlib.sha1(raw).hexdigest()[:12]

    def _paper_content(self, paper: Dict[str, Any]) -> Tuple[str, str]:
        expert = paper.get("expert_level_deep_analysis", {})
        if isinstance(expert, dict):
            text = str(expert.get("final_deep_analysis", "") or "")
            if text.strip():
                return "full_text_analysis", text
        for field, level in [
            ("query_centered_content", "full_text_analysis"),
            ("abstract_note", "abstract_level"),
            ("research_content", "corpus_summary_level"),
        ]:
            text = str(paper.get(field, "") or "")
            if text.strip():
                return level, text
        return "metadata_only", ""

    def _local_evidence(
        self,
        question: str,
        gap: Dict[str, Any],
        local_k: int,
        content_source: str,
        survey_mode: str,
    ) -> List[Dict[str, Any]]:
        query = str(gap.get("search_query", "") or question).strip()
        if self.retrieval_agent is None:
            self.retrieval_agent = RetrievalAgent.from_config()
        if self.relevance_agent is None:
            self.relevance_agent = RelevanceAgent(temperature=0.0)

        papers = self.retrieval_agent.retrieve(query=query, top_k=local_k, normalize=True)
        selected = self.relevance_agent.screen_top_m(
            query=question,
            papers=papers,
            batch_size=min(5, max(1, local_k)),
        )

        if gap.get("reading_level") == "full_text" and selected:
            digest_agent = PaperDigestAgent(
                temperature=0.1,
                max_pdf_chars=DEFAULT_MAX_SOURCE_CHARS,
                content_source=content_source,
                use_cache=False,
            )
            selected = digest_agent.build_middle_records(
                query=question,
                selected_papers=selected,
            )

        evidence = []
        for paper in selected:
            reading_level, content = self._paper_content(paper)
            if not content.strip():
                continue
            source_key = str(
                paper.get("paper_id") or paper.get("file_name") or paper.get("title")
            )
            evidence.append(
                {
                    "evidence_id": self._id("local_literature", source_key, gap["gap_id"]),
                    "gap_id": gap["gap_id"],
                    "source_type": "local_literature",
                    "source_id": source_key,
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "authors": paper.get("authors", ""),
                    "journal": paper.get("journal", ""),
                    "reading_level": reading_level,
                    "content": content,
                    "retrieval_query": query,
                    "limitations": (
                        ["Evidence was inspected below full-text analysis level."]
                        if reading_level != "full_text_analysis"
                        else []
                    ),
                }
            )
        return evidence

    def _web_evidence(
        self,
        question: str,
        gap: Dict[str, Any],
        web_k: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        queries = [str(value or "").strip() for value in gap.get("web_queries", [])]
        queries = [value for value in queries if value]
        if not queries:
            query = str(gap.get("search_query", "") or question).strip()
            queries = [query]

        searcher = WebLiteratureSearcher(
            ads_api_token=SUPPLEMENT_ADS_API_TOKEN,
            timeout=SUPPLEMENT_WEB_TIMEOUT,
        )
        output = searcher.search_many(
            queries=queries,
            engines=SUPPLEMENT_WEB_ENGINES,
            limit_per_query=max(1, web_k // max(1, len(queries))),
            from_year=SUPPLEMENT_FROM_YEAR,
        )
        candidates = output.get("results", [])[:web_k]
        if not candidates:
            return [], output.get("errors", [])

        if self.supplement_agent is None:
            self.supplement_agent = SupplementAgent(temperature=0.0)
        plan = {
            "should_supplement": True,
            "reason": gap.get("why_gap", ""),
            "gaps": [gap],
            "mode": "discovery",
        }
        screening = self.supplement_agent.screen_supplement_candidates(
            query=question,
            survey_mode="discovery",
            supplement_plan=plan,
            local_candidates=[],
            web_candidates=candidates,
            accept_relevance_levels=SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS,
        )

        evidence = []
        for paper in screening.get("accepted_web_candidates", []):
            content = str(paper.get("abstract", "") or "").strip()
            if not content:
                continue
            source_key = str(
                paper.get("doi") or paper.get("external_id") or paper.get("title")
            )
            evidence.append(
                {
                    "evidence_id": self._id("web_literature", source_key, gap["gap_id"]),
                    "gap_id": gap["gap_id"],
                    "source_type": "web_literature",
                    "source_id": source_key,
                    "source": paper.get("source", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "authors": paper.get("authors", ""),
                    "journal": paper.get("journal", ""),
                    "doi": paper.get("doi", ""),
                    "url": paper.get("url", ""),
                    "reading_level": "abstract_level",
                    "content": content,
                    "retrieval_query": paper.get("query", ""),
                    "limitations": [
                        "External evidence was inspected at abstract level only."
                    ],
                }
            )
        return evidence, output.get("errors", [])

    def _geo_evidence(
        self,
        gap: Dict[str, Any],
        coordinate: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        context, summary = query_and_summarize_geo_context(
            lat=float(coordinate["lat"]),
            lon=float(coordinate.get("lon_180", coordinate["lon"])),
        )
        errors = [
            {"source": key, "message": value}
            for key, value in (context.get("errors", {}) or {}).items()
        ]
        if not summary.strip():
            return [], errors
        source_key = f"{coordinate['lat']}|{coordinate.get('lon_180', coordinate['lon'])}"
        return [
            {
                "evidence_id": self._id("geo_product", source_key, gap["gap_id"]),
                "gap_id": gap["gap_id"],
                "source_type": "geo_product",
                "source_id": source_key,
                "reading_level": "data_product_summary",
                "content": summary,
                "required_data": gap.get("required_data", []),
                "location": coordinate,
                "product_errors": context.get("errors", {}),
                "limitations": [
                    "Product availability and resolution differ across the queried location."
                ],
            }
        ], errors

    def acquire(
        self,
        question: str,
        gaps: List[Dict[str, Any]],
        coordinate: Optional[Dict[str, Any]],
        allowed_sources: List[str],
        local_k: int = 5,
        web_k: int = 5,
        content_source: str = "md",
        survey_mode: str = "full",
    ) -> Dict[str, Any]:
        allowed = {str(value).lower() for value in allowed_sources}
        evidence: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        tasks: List[Dict[str, Any]] = []

        for gap in gaps:
            source = str(gap.get("recommended_source", "none") or "none").lower()
            task = {
                "gap_id": gap.get("gap_id", ""),
                "source": source,
                "query": gap.get("search_query", ""),
                "status": "pending",
            }
            try:
                if source == "local" and source in allowed:
                    acquired = self._local_evidence(
                        question, gap, local_k, content_source, survey_mode
                    )
                    evidence.extend(acquired)
                elif source == "web" and source in allowed:
                    acquired, source_errors = self._web_evidence(question, gap, web_k)
                    evidence.extend(acquired)
                    errors.extend(source_errors)
                elif source == "geo" and source in allowed and coordinate:
                    acquired, source_errors = self._geo_evidence(gap, coordinate)
                    evidence.extend(acquired)
                    errors.extend(source_errors)
                else:
                    acquired = []
                task["status"] = "success" if acquired else "no_evidence"
                task["evidence_count"] = len(acquired)
            except Exception as exc:
                task["status"] = "failed"
                task["evidence_count"] = 0
                errors.append(
                    {
                        "gap_id": gap.get("gap_id", ""),
                        "source": source,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            tasks.append(task)

        deduped = []
        seen = set()
        for item in evidence:
            evidence_id = item["evidence_id"]
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            deduped.append(item)
        return {"evidence": deduped, "tasks": tasks, "errors": errors}
