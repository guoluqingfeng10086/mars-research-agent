import json
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from config import (
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
)

from lit_agent.prompts.supplement_prompts import (
    SUPPLEMENT_DECISION_SYSTEM_PROMPT,
    SUPPLEMENT_DECISION_USER_PROMPT,
    SUPPLEMENT_RELEVANCE_SYSTEM_PROMPT,
    SUPPLEMENT_RELEVANCE_USER_PROMPT,
    SUPPLEMENT_MINDMAP_INTEGRATION_PROMPT,
    SUPPLEMENT_SURVEY_INTEGRATION_PROMPT,
    SUPPLEMENT_SYNTHESIS_SYSTEM_PROMPT,
    SUPPLEMENT_SYNTHESIS_USER_PROMPT,
)


class SupplementAgent:
    """
    Optional coverage-audit agent for the experimental supplement flow.

    This agent does not mutate selected papers or paper records. It only:
    1. decides whether one small supplementary evidence round is useful;
    2. generates local/web supplement queries;
    3. synthesizes supplementary candidates into a side report.
    """

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        temperature: float = 0.0,
        max_record_chars: int = 1800,
        max_candidate_chars: int = 900,
    ):
        self.model = model
        self.temperature = temperature
        self.max_record_chars = max_record_chars
        self.max_candidate_chars = max_candidate_chars

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _strip_json_markdown(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            return text

        if text.startswith("```"):
            text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            return text

        return text

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        text = self._strip_json_markdown(response)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON during supplement analysis.\n"
                f"Raw response:\n{response}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Expected a JSON object during supplement analysis, "
                f"but got: {type(data)}"
            )

        return data

    def _parse_json_list(self, response: str) -> List[Dict[str, Any]]:
        text = self._strip_json_markdown(response)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON during supplement screening.\n"
                f"Raw response:\n{response}"
            ) from exc

        if isinstance(data, dict) and "results" in data:
            data = data["results"]

        if not isinstance(data, list):
            raise ValueError(
                "Expected a JSON list during supplement screening, "
                f"but got: {type(data)}"
            )

        return data

    def _truncate(self, text: Any, limit: int) -> str:
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _relevance_level(self, paper: Dict[str, Any]) -> str:
        return str(
            paper.get("screening_result", {}).get("relevance_level", "")
        ).lower()

    def _record_content(self, paper: Dict[str, Any]) -> str:
        if paper.get("record_source") == "expert_level_deep_analysis":
            expert_obj = paper.get("expert_level_deep_analysis", {})
            if isinstance(expert_obj, dict):
                return expert_obj.get("final_deep_analysis", "") or ""

        for field in [
            "query_centered_content",
            "abstract_note",
            "research_content",
            "abstract",
        ]:
            value = paper.get(field, "")
            if str(value or "").strip():
                return str(value)

        return ""

    def _compact_paper_records(
        self,
        paper_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        compact = []

        for paper in paper_records:
            compact.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "authors": paper.get("authors", ""),
                    "research_region": paper.get("research_region", ""),
                    "relevance_level": self._relevance_level(paper),
                    "record_source": paper.get("record_source", ""),
                    "record_content": self._truncate(
                        self._record_content(paper),
                        self.max_record_chars,
                    ),
                }
            )

        return compact

    def _compact_local_candidates(
        self,
        papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        compact = []

        for paper in papers:
            source_read, reading_level, content = self._candidate_content(
                paper=paper,
                source_type="local",
            )
            compact.append(
                {
                    "source_type": "local",
                    "paper_id": paper.get("paper_id", ""),
                    "rank": paper.get("rank"),
                    "score": paper.get("score"),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "authors": paper.get("authors", ""),
                    "research_region": paper.get("research_region", ""),
                    "source_read": source_read,
                    "reading_level": reading_level,
                    "content": self._truncate(content, self.max_candidate_chars),
                }
            )

        return compact

    def _compact_web_candidates(
        self,
        papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        compact = []

        for paper in papers:
            source_read, reading_level, content = self._candidate_content(
                paper=paper,
                source_type="web",
            )
            compact.append(
                {
                    "source_type": "web",
                    "source": paper.get("source", ""),
                    "query": paper.get("query", ""),
                    "rank": paper.get("rank"),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "authors": paper.get("authors", ""),
                    "doi": paper.get("doi", ""),
                    "url": paper.get("url", ""),
                    "citation_count": paper.get("citation_count"),
                    "source_read": source_read,
                    "reading_level": reading_level,
                    "content": self._truncate(content, self.max_candidate_chars),
                }
            )

        return compact

    def _candidate_content(
        self,
        paper: Dict[str, Any],
        source_type: str,
    ) -> Tuple[str, str, str]:
        if source_type == "local":
            for field, reading_level in [
                ("abstract_note", "abstract_level"),
                ("research_content", "corpus_summary_level"),
                ("abstract", "abstract_level"),
            ]:
                content = str(paper.get(field, "") or "").strip()
                if content:
                    return field, reading_level, content
        else:
            content = str(paper.get("abstract", "") or "").strip()
            if content:
                return "abstract", "abstract_level", content

        return "metadata_only", "metadata_only", ""

    def _candidate_citation(self, candidate: Dict[str, Any]) -> str:
        authors = candidate.get("authors", "")
        year = str(candidate.get("year", "") or "").strip()
        title = str(candidate.get("title", "") or "").strip()

        if isinstance(authors, list):
            author_text = str(authors[0] if authors else "").strip()
            if len(authors) > 1 and author_text:
                author_text += " et al."
        else:
            author_text = str(authors or "").strip()

        if author_text and year:
            return f"{author_text} ({year})"
        if title and year:
            return f"{title} ({year})"
        return title or "Supplementary candidate"

    def _normalize_synthesis_report(
        self,
        report: Dict[str, Any],
        local_candidates: List[Dict[str, Any]],
        web_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        compact_candidates = (
            self._compact_local_candidates(local_candidates)
            + self._compact_web_candidates(web_candidates)
        )
        candidate_by_title = {
            str(item.get("title", "") or "").strip().lower(): item
            for item in compact_candidates
            if str(item.get("title", "") or "").strip()
        }

        normalized_cards = []
        raw_cards = report.get("accepted_supplementary_papers", [])
        if not isinstance(raw_cards, list):
            raw_cards = []

        for raw_card in raw_cards:
            if not isinstance(raw_card, dict):
                continue

            title_key = str(raw_card.get("title", "") or "").strip().lower()
            candidate = candidate_by_title.get(title_key)
            if not candidate or candidate.get("source_read") == "metadata_only":
                continue

            card = dict(raw_card)
            card["citation_id"] = f"S{len(normalized_cards) + 1}"
            card["citation"] = str(card.get("citation", "") or "").strip() or (
                self._candidate_citation(candidate)
            )
            for field in [
                "title",
                "year",
                "journal",
                "source_type",
                "source_read",
                "reading_level",
            ]:
                card[field] = candidate.get(field, "")
            normalized_cards.append(card)

        report["evidence_level"] = "abstract_or_summary_level_supplement"
        report["accepted_supplementary_papers"] = normalized_cards
        report.setdefault("supplement_used", bool(normalized_cards))
        report.setdefault("overall_effect", "add_context" if normalized_cards else "none")
        if not normalized_cards:
            report["supplement_used"] = False
            report["overall_effect"] = "none"
            report["integration_analysis"] = []

        valid_ids = {card["citation_id"] for card in normalized_cards}
        integration = report.get("integration_analysis", [])
        if not isinstance(integration, list):
            integration = []
        normalized_integration = []
        for item in integration:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item["citations"] = [
                citation_id
                for citation_id in item.get("citations", [])
                if citation_id in valid_ids
            ]
            if item["citations"]:
                normalized_integration.append(item)
        report["integration_analysis"] = normalized_integration
        report.setdefault("follow_up_candidates", [])
        return report

    def _dedupe_queries(self, queries: List[str], limit: int) -> List[str]:
        seen = set()
        out = []

        for query in queries:
            query = str(query or "").strip()
            key = query.lower()
            if not query or key in seen:
                continue
            seen.add(key)
            out.append(query)
            if len(out) >= limit:
                break

        return out

    def _build_screening_candidates(
        self,
        local_candidates: List[Dict[str, Any]],
        web_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates = []

        for index, paper in enumerate(local_candidates):
            source_read, reading_level, content = self._candidate_content(
                paper=paper,
                source_type="local",
            )
            candidates.append(
                {
                    "row_id": len(candidates),
                    "source_type": "local",
                    "source_index": index,
                    "retrieval_query": paper.get("supplement_query", ""),
                    "rank": paper.get("rank"),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "authors": paper.get("authors", ""),
                    "research_region": paper.get("research_region", ""),
                    "source_read": source_read,
                    "reading_level": reading_level,
                    "content": self._truncate(content, self.max_candidate_chars),
                }
            )

        for index, paper in enumerate(web_candidates):
            source_read, reading_level, content = self._candidate_content(
                paper=paper,
                source_type="web",
            )
            candidates.append(
                {
                    "row_id": len(candidates),
                    "source_type": "web",
                    "source_index": index,
                    "source": paper.get("source", ""),
                    "retrieval_query": paper.get("query", ""),
                    "rank": paper.get("rank"),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "authors": paper.get("authors", ""),
                    "doi": paper.get("doi", ""),
                    "source_read": source_read,
                    "reading_level": reading_level,
                    "content": self._truncate(content, self.max_candidate_chars),
                }
            )

        return candidates

    def screen_supplement_candidates(
        self,
        query: str,
        survey_mode: str,
        supplement_plan: Dict[str, Any],
        local_candidates: List[Dict[str, Any]],
        web_candidates: List[Dict[str, Any]],
        accept_relevance_levels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        screening_candidates = self._build_screening_candidates(
            local_candidates=local_candidates,
            web_candidates=web_candidates,
        )

        if not screening_candidates:
            return {
                "enabled": True,
                "screening_results": [],
                "accepted_local_candidates": [],
                "accepted_web_candidates": [],
                "rejected_count": 0,
            }

        accept_levels = {
            str(level).lower()
            for level in (accept_relevance_levels or ["high", "medium"])
        }

        user_prompt = SUPPLEMENT_RELEVANCE_USER_PROMPT.format(
            query=query,
            survey_mode=survey_mode,
            supplement_plan_json=json.dumps(
                supplement_plan,
                ensure_ascii=False,
                indent=2,
            ),
            candidates_json=json.dumps(
                screening_candidates,
                ensure_ascii=False,
                indent=2,
            ),
        )

        response = self._call_llm(
            system_prompt=SUPPLEMENT_RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        raw_results = self._parse_json_list(response)

        result_by_row_id = {}
        for item in raw_results:
            try:
                row_id = int(item.get("row_id"))
            except Exception:
                continue
            result_by_row_id[row_id] = item

        accepted_local = []
        accepted_web = []
        screening_results = []

        for candidate in screening_candidates:
            row_id = candidate["row_id"]
            result = result_by_row_id.get(row_id)

            if result is None:
                result = {
                    "row_id": row_id,
                    "relevance_level": "unknown",
                    "is_usable": False,
                    "matched_gap_topic": "",
                    "reason": "No valid screening result was returned.",
                }

            relevance_level = str(
                result.get("relevance_level", "unknown")
            ).lower()
            is_usable = bool(result.get("is_usable"))
            source_read = candidate.get("source_read", "metadata_only")
            has_read_content = source_read != "metadata_only"
            is_accepted = (
                is_usable
                and relevance_level in accept_levels
                and has_read_content
            )

            normalized_result = {
                "row_id": row_id,
                "source_type": candidate.get("source_type", ""),
                "source_index": candidate.get("source_index"),
                "title": candidate.get("title", ""),
                "source_read": source_read,
                "reading_level": candidate.get("reading_level", "metadata_only"),
                "relevance_level": relevance_level,
                "is_usable": is_usable,
                "is_accepted": is_accepted,
                "matched_gap_topic": result.get("matched_gap_topic", ""),
                "reason": result.get("reason", ""),
            }
            screening_results.append(normalized_result)

            if not is_accepted:
                continue

            source_index = candidate.get("source_index")
            if candidate.get("source_type") == "local":
                accepted = dict(local_candidates[source_index])
                accepted["supplement_screening"] = normalized_result
                accepted_local.append(accepted)
            elif candidate.get("source_type") == "web":
                accepted = dict(web_candidates[source_index])
                accepted["supplement_screening"] = normalized_result
                accepted_web.append(accepted)

        return {
            "enabled": True,
            "accept_relevance_levels": sorted(accept_levels),
            "screening_results": screening_results,
            "accepted_local_candidates": accepted_local,
            "accepted_web_candidates": accepted_web,
            "rejected_count": len(screening_candidates)
            - len(accepted_local)
            - len(accepted_web),
        }

    def _normalize_plan(
        self,
        plan: Dict[str, Any],
        query: str,
        mode: str,
        max_gaps: int,
        max_web_queries_per_gap: int,
    ) -> Dict[str, Any]:
        should_supplement = bool(plan.get("should_supplement"))
        gaps = plan.get("gaps", [])
        if not isinstance(gaps, list):
            gaps = []

        normalized_gaps = []
        for gap in gaps[:max_gaps]:
            if not isinstance(gap, dict):
                continue

            web_queries = gap.get("web_queries", [])
            if not isinstance(web_queries, list):
                web_queries = []

            local_query = str(gap.get("local_query", "") or "").strip()
            if not local_query:
                local_query = query

            normalized_gaps.append(
                {
                    "topic": str(gap.get("topic", "") or "").strip(),
                    "gap_type": str(gap.get("gap_type", "") or "").strip(),
                    "why_gap": str(gap.get("why_gap", "") or "").strip(),
                    "local_query": local_query,
                    "web_queries": self._dedupe_queries(
                        web_queries,
                        max_web_queries_per_gap,
                    ),
                }
            )

        if mode == "force" and not normalized_gaps:
            should_supplement = True
            normalized_gaps = [
                {
                    "topic": "general coverage check",
                    "gap_type": "sparse_local_coverage",
                    "why_gap": "Supplement mode is forced by configuration.",
                    "local_query": query,
                    "web_queries": [query],
                }
            ]

        if mode == "force":
            should_supplement = True

        if not normalized_gaps:
            should_supplement = False

        return {
            "should_supplement": should_supplement,
            "reason": str(plan.get("reason", "") or "").strip(),
            "gaps": normalized_gaps,
            "mode": mode,
        }

    def plan_supplement(
        self,
        query: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
        mode: str = "auto",
        max_gaps: int = 2,
        max_web_queries_per_gap: int = 2,
    ) -> Dict[str, Any]:
        mode = str(mode or "auto").lower()

        if mode == "off":
            return {
                "should_supplement": False,
                "reason": "Supplement mode is off.",
                "gaps": [],
                "mode": mode,
            }

        compact_records = self._compact_paper_records(paper_records)

        user_prompt = SUPPLEMENT_DECISION_USER_PROMPT.format(
            query=query,
            survey_mode=survey_mode,
            paper_records_json=json.dumps(
                compact_records,
                ensure_ascii=False,
                indent=2,
            ),
            max_gaps=max_gaps,
            max_web_queries_per_gap=max_web_queries_per_gap,
        )

        response = self._call_llm(
            system_prompt=SUPPLEMENT_DECISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        plan = self._parse_json_object(response)

        return self._normalize_plan(
            plan=plan,
            query=query,
            mode=mode,
            max_gaps=max_gaps,
            max_web_queries_per_gap=max_web_queries_per_gap,
        )

    def synthesize_supplement(
        self,
        query: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
        supplement_plan: Dict[str, Any],
        local_candidates: List[Dict[str, Any]],
        web_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not local_candidates and not web_candidates:
            return {
                "supplement_used": False,
                "overall_effect": "none",
                "evidence_level": "abstract_or_summary_level_supplement",
                "summary": "No supplementary candidates were found.",
                "accepted_supplementary_papers": [],
                "integration_analysis": [],
                "follow_up_candidates": [],
            }

        user_prompt = SUPPLEMENT_SYNTHESIS_USER_PROMPT.format(
            query=query,
            survey_mode=survey_mode,
            supplement_plan_json=json.dumps(
                supplement_plan,
                ensure_ascii=False,
                indent=2,
            ),
            paper_records_json=json.dumps(
                self._compact_paper_records(paper_records),
                ensure_ascii=False,
                indent=2,
            ),
            local_candidates_json=json.dumps(
                self._compact_local_candidates(local_candidates),
                ensure_ascii=False,
                indent=2,
            ),
            web_candidates_json=json.dumps(
                self._compact_web_candidates(web_candidates),
                ensure_ascii=False,
                indent=2,
            ),
        )

        response = self._call_llm(
            system_prompt=SUPPLEMENT_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        report = self._parse_json_object(response)
        report = self._normalize_synthesis_report(
            report=report,
            local_candidates=local_candidates,
            web_candidates=web_candidates,
        )
        report["supplement_plan"] = supplement_plan
        return report

    def format_report_for_survey(
        self,
        supplement_report: Optional[Dict[str, Any]],
    ) -> str:
        if not supplement_report or not supplement_report.get("supplement_used"):
            return ""

        return "\n\n" + SUPPLEMENT_SURVEY_INTEGRATION_PROMPT.format(
            supplement_report_json=json.dumps(
                self._compact_report_for_integration(supplement_report),
                ensure_ascii=False,
                indent=2,
            )
        )

    def format_report_for_mindmap(
        self,
        supplement_report: Optional[Dict[str, Any]],
    ) -> str:
        if not supplement_report or not supplement_report.get("supplement_used"):
            return ""

        return "\n\n" + SUPPLEMENT_MINDMAP_INTEGRATION_PROMPT.format(
            supplement_report_json=json.dumps(
                self._compact_report_for_integration(supplement_report),
                ensure_ascii=False,
                indent=2,
            )
        )

    def _compact_report_for_integration(
        self,
        supplement_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "evidence_level": supplement_report.get(
                "evidence_level",
                "abstract_or_summary_level_supplement",
            ),
            "overall_effect": supplement_report.get("overall_effect", "none"),
            "summary": supplement_report.get("summary", ""),
            "accepted_supplementary_papers": supplement_report.get(
                "accepted_supplementary_papers",
                [],
            ),
            "integration_analysis": supplement_report.get("integration_analysis", []),
            "follow_up_candidates": supplement_report.get("follow_up_candidates", []),
        }

    def format_report_as_markdown(
        self,
        supplement_report: Optional[Dict[str, Any]],
    ) -> str:
        report = supplement_report or {}
        used = bool(report.get("supplement_used"))
        lines = [
            "# Supplementary Evidence Report",
            "",
            "## Status",
            "",
            f"- Used in final synthesis: {'yes' if used else 'no'}",
            f"- Evidence level: {report.get('evidence_level', 'not applicable')}",
            f"- Overall effect: {report.get('overall_effect', 'none')}",
            f"- Summary: {report.get('summary', 'No supplementary evidence used.')}",
            "",
        ]

        accepted = report.get("accepted_supplementary_papers", [])
        if accepted:
            lines.extend(["## Accepted Supplementary Evidence", ""])
            for paper in accepted:
                citation_id = paper.get("citation_id", "S?")
                citation = paper.get("citation", "")
                title = paper.get("title", "")
                lines.extend(
                    [
                        f"### {citation_id}. {citation}",
                        "",
                        f"- Title: {title}",
                        f"- Reading level: {paper.get('reading_level', '')}",
                        f"- Source read: {paper.get('source_read', '')}",
                        f"- Why selected: {paper.get('why_selected', '')}",
                        f"- Distilled finding: {paper.get('distilled_finding', '')}",
                        "- Relation to main survey: "
                        f"{paper.get('relation_to_main_survey', '')}",
                        f"- Confidence: {paper.get('confidence', '')}",
                        f"- Limitations: {paper.get('limitations', '')}",
                        "",
                    ]
                )

        integration = report.get("integration_analysis", [])
        if integration:
            lines.extend(["## Integration With Main Survey", ""])
            for item in integration:
                citations = ", ".join(item.get("citations", []))
                lines.extend(
                    [
                        f"- Main claim: {item.get('main_claim', '')}",
                        f"  Effect: {item.get('effect', '')}. "
                        f"{item.get('reason', '')} [{citations}]",
                    ]
                )
            lines.append("")

        follow_up = report.get("follow_up_candidates", [])
        if follow_up:
            lines.extend(["## Candidates For Full-Text Follow-Up", ""])
            for item in follow_up:
                lines.append(
                    f"- {item.get('title', '')} ({item.get('year', '')}): "
                    f"{item.get('why_follow_up', '')}"
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
