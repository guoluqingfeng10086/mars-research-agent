import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from config import QUERY_LLM_API_KEY, QUERY_LLM_BASE_URL, QUERY_LLM_MODEL
from lit_agent.prompts.discovery_prompts import (
    RESEARCH_MAP_SYSTEM_PROMPT,
    RESEARCH_MAP_USER_PROMPT,
)
from lit_agent.utils.io_utils import save_json


class ResearchMapAgent:
    """Build and incrementally extend a machine-readable research map."""

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        temperature: float = 0.0,
        client=None,
        max_record_chars: int = 6000,
        max_context_chars: int = 45000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_record_chars = max_record_chars
        self.max_context_chars = max_context_chars
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

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

    def _parse_json(self, response: str) -> Dict[str, Any]:
        text = str(response or "").strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*|\s*```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("Research map LLM returned invalid JSON.") from exc
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as nested_exc:
                raise ValueError("Research map LLM returned invalid JSON.") from nested_exc
        if not isinstance(data, dict):
            raise ValueError("Research map must be a JSON object.")
        return data

    def _record_content(self, paper: Dict[str, Any]) -> Tuple[str, str]:
        expert = paper.get("expert_level_deep_analysis", {})
        if isinstance(expert, dict):
            content = str(expert.get("final_deep_analysis", "") or "")
            if content.strip():
                return "expert_level_deep_analysis", content
        for field in ["query_centered_content", "abstract_note", "research_content"]:
            content = str(paper.get(field, "") or "")
            if content.strip():
                return field, content
        return "metadata_only", ""

    def _compact_records(self, paper_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for paper in paper_records:
            source, content = self._record_content(paper)
            compact.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "authors": paper.get("authors", ""),
                    "journal": paper.get("journal", ""),
                    "file_name": paper.get("file_name", ""),
                    "research_region": paper.get("research_region", ""),
                    "relevance_level": paper.get("screening_result", {}).get(
                        "relevance_level", ""
                    ),
                    "record_source": source,
                    "record_content": content[: self.max_record_chars],
                }
            )
        return compact

    def empty_map(self, question: str) -> Dict[str, Any]:
        return {
            "version": 1,
            "question": question,
            "central_hypotheses": [],
            "claims": [],
            "evidence": [],
            "alternatives": [],
            "gaps": [],
            "source_ids": [],
            "acquired_evidence": [],
        }

    def _normalize(self, data: Dict[str, Any], question: str) -> Dict[str, Any]:
        normalized = self.empty_map(question)
        normalized["question"] = str(data.get("question", "") or question)
        for field in [
            "central_hypotheses",
            "claims",
            "evidence",
            "alternatives",
            "gaps",
            "source_ids",
            "acquired_evidence",
        ]:
            value = data.get(field, [])
            normalized[field] = value if isinstance(value, list) else []

        claim_ids = set()
        claims = []
        for index, raw in enumerate(normalized["claims"], start=1):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            claim_id = str(item.get("claim_id", "") or f"C{index}")
            if claim_id in claim_ids:
                claim_id = f"C{index}"
            claim_ids.add(claim_id)
            item["claim_id"] = claim_id
            claims.append(item)
        normalized["claims"] = claims

        evidence_ids = set()
        evidence = []
        for index, raw in enumerate(normalized["evidence"], start=1):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            evidence_id = str(item.get("evidence_id", "") or f"E{index}")
            if evidence_id in evidence_ids:
                evidence_id = f"E{index}"
            evidence_ids.add(evidence_id)
            item["evidence_id"] = evidence_id
            linked = item.get("claim_ids", [])
            item["claim_ids"] = [value for value in linked if value in claim_ids]
            evidence.append(item)
        normalized["evidence"] = evidence
        normalized["source_ids"] = sorted(
            {
                str(item.get("source_id", "") or "").strip()
                for item in evidence
                if str(item.get("source_id", "") or "").strip()
            }
        )
        return normalized

    def generate(
        self,
        question: str,
        survey_text: str,
        mindmap_text: str,
        paper_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        user_prompt = RESEARCH_MAP_USER_PROMPT.format(
            question=question,
            survey_text=str(survey_text or "")[: self.max_context_chars],
            mindmap_text=str(mindmap_text or "")[: self.max_context_chars],
            paper_records_json=json.dumps(
                self._compact_records(paper_records), ensure_ascii=False, indent=2
            ),
        )
        response = self._call_llm(RESEARCH_MAP_SYSTEM_PROMPT, user_prompt)
        return self._normalize(self._parse_json(response), question)

    def generate_and_save(
        self,
        question: str,
        survey_text: str,
        mindmap_text: str,
        paper_records: List[Dict[str, Any]],
        output_path,
    ) -> Dict[str, Any]:
        research_map = self.generate(
            question=question,
            survey_text=survey_text,
            mindmap_text=mindmap_text,
            paper_records=paper_records,
        )
        save_json(research_map, output_path)
        return research_map

    def update(
        self,
        research_map: Dict[str, Any],
        new_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Append traceable round evidence; the next gap pass interprets its effect."""
        updated = dict(research_map or {})
        existing = list(updated.get("acquired_evidence", []))
        seen = {
            str(item.get("evidence_id", "") or "")
            for item in existing
            if isinstance(item, dict)
        }
        for item in new_evidence:
            evidence_id = str(item.get("evidence_id", "") or "")
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            existing.append(item)
        updated["acquired_evidence"] = existing
        updated["version"] = int(updated.get("version", 1) or 1) + 1
        return updated

    def build_question_subgraph(self, research_map: Dict[str, Any]) -> Dict[str, Any]:
        """Traverse claim dependencies and return the evidence-bearing question graph."""
        claims = {
            str(item.get("claim_id")): item
            for item in research_map.get("claims", [])
            if isinstance(item, dict) and item.get("claim_id")
        }
        roots = [
            claim_id
            for claim_id, item in claims.items()
            if item.get("importance") == "core"
        ] or list(claims)

        visited = set()
        ordered_claim_ids = []

        def visit(claim_id: str) -> None:
            if claim_id in visited or claim_id not in claims:
                return
            visited.add(claim_id)
            ordered_claim_ids.append(claim_id)
            dependencies = claims[claim_id].get("depends_on_claim_ids", [])
            if isinstance(dependencies, list):
                for dependency in dependencies:
                    visit(str(dependency))

        for root in roots:
            visit(root)

        evidence = []
        for item in research_map.get("evidence", []):
            if not isinstance(item, dict):
                continue
            linked = item.get("claim_ids", [])
            if not linked or any(str(claim_id) in visited for claim_id in linked):
                evidence.append(item)

        alternatives = []
        for item in research_map.get("alternatives", []):
            if not isinstance(item, dict):
                continue
            linked = item.get("related_claim_ids", [])
            if not linked or any(str(claim_id) in visited for claim_id in linked):
                alternatives.append(item)

        return {
            "version": research_map.get("version", 1),
            "question": research_map.get("question", ""),
            "central_hypotheses": research_map.get("central_hypotheses", []),
            "claims": [claims[claim_id] for claim_id in ordered_claim_ids],
            "evidence": evidence,
            "alternatives": alternatives,
            "gaps": research_map.get("gaps", []),
            "source_ids": research_map.get("source_ids", []),
            "acquired_evidence": research_map.get("acquired_evidence", []),
        }
