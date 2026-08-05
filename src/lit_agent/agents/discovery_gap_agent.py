import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import QUERY_LLM_API_KEY, QUERY_LLM_BASE_URL, QUERY_LLM_MODEL
from lit_agent.prompts.discovery_prompts import (
    DISCOVERY_GAP_SYSTEM_PROMPT,
    DISCOVERY_GAP_USER_PROMPT,
)


class DiscoveryGapAgent:
    """Evaluate answer completeness and plan a bounded set of evidence gaps."""

    VALID_SOURCES = {"local", "web", "geo", "none"}

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        temperature: float = 0.0,
        client=None,
    ):
        self.model = model
        self.temperature = temperature
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
                raise ValueError("Discovery completeness LLM returned invalid JSON.") from exc
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Discovery completeness result must be a JSON object.")
        return data

    def _score(self, value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0

    def _normalize(
        self,
        data: Dict[str, Any],
        allowed_sources: List[str],
        coordinate: Optional[Dict[str, Any]],
        max_gaps: int,
    ) -> Dict[str, Any]:
        allowed = {str(value).lower() for value in allowed_sources}
        raw_scores = data.get("completeness", {})
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        scores = {
            key: self._score(raw_scores.get(key, 0.0))
            for key in [
                "coverage",
                "directness",
                "source_quality",
                "spatial_fit",
                "consistency",
                "alternative_coverage",
            ]
        }

        gaps = []
        for index, raw in enumerate(data.get("gaps", []) or [], start=1):
            if not isinstance(raw, dict) or len(gaps) >= max_gaps:
                continue
            item = dict(raw)
            source = str(item.get("recommended_source", "none") or "none").lower()
            if source not in self.VALID_SOURCES or source not in allowed:
                source = "none"
            if source == "geo" and not coordinate:
                source = "none"
            web_queries = item.get("web_queries", [])
            item.update(
                {
                    "gap_id": str(item.get("gap_id", "") or f"G{index}"),
                    "claim_ids": item.get("claim_ids", [])
                    if isinstance(item.get("claim_ids", []), list)
                    else [],
                    "priority": str(item.get("priority", "medium") or "medium").lower(),
                    "recommended_source": source,
                    "search_query": str(item.get("search_query", "") or "").strip(),
                    "web_queries": web_queries if isinstance(web_queries, list) else [],
                    "required_data": item.get("required_data", [])
                    if isinstance(item.get("required_data", []), list)
                    else [],
                    "reading_level": str(
                        item.get("reading_level", "summary") or "summary"
                    ).lower(),
                }
            )
            gaps.append(item)

        answerability = str(data.get("answerability", "insufficient") or "insufficient")
        if answerability not in {"sufficient", "qualified", "insufficient"}:
            answerability = "insufficient"
        is_complete = bool(data.get("is_complete"))
        if is_complete:
            gaps = [gap for gap in gaps if gap.get("priority") == "low"]

        return {
            "is_complete": is_complete,
            "answerability": answerability,
            "reason": str(data.get("reason", "") or ""),
            "completeness": scores,
            "gaps": gaps,
        }

    def analyze(
        self,
        question: str,
        research_map: Dict[str, Any],
        acquired_evidence: List[Dict[str, Any]],
        allowed_sources: List[str],
        coordinate: Optional[Dict[str, Any]],
        max_gaps: int = 3,
    ) -> Dict[str, Any]:
        user_prompt = DISCOVERY_GAP_USER_PROMPT.format(
            question=question,
            allowed_sources_json=json.dumps(allowed_sources, ensure_ascii=False),
            coordinate_json=json.dumps(coordinate, ensure_ascii=False, indent=2),
            research_map_json=json.dumps(research_map, ensure_ascii=False, indent=2),
            acquired_evidence_json=json.dumps(
                acquired_evidence, ensure_ascii=False, indent=2
            ),
            max_gaps=max_gaps,
        )
        response = self._call_llm(DISCOVERY_GAP_SYSTEM_PROMPT, user_prompt)
        return self._normalize(
            self._parse_json(response),
            allowed_sources=allowed_sources,
            coordinate=coordinate,
            max_gaps=max_gaps,
        )
