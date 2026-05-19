# src/lit_agent/agents/mindmap_agent.py

import json
from typing import Any, Dict, List

from openai import OpenAI

from config import (
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
)

from lit_agent.prompts.survey_prompts import (
    FINAL_MINDMAP_SYSTEM_PROMPT,
    FINAL_MINDMAP_USER_PROMPT,
)
from lit_agent.utils.io_utils import write_text


class MindMapAgent:
    """
    Generate final MindMap.txt.

    lightweight:
        LLM input uses abstract_note or research_content.

    middle:
        LLM input uses query_centered_content.

    full:
        high papers use expert_level_deep_analysis.final_deep_analysis.
        medium papers use query_centered_content.
    """

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        temperature: float = 0.2,
    ):
        self.model = model
        self.temperature = temperature
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

    def _safe_year(self, item: Dict[str, Any]) -> int:
        try:
            return int(float(item.get("year", 9999)))
        except Exception:
            return 9999

    def _relevance_level(self, paper: Dict[str, Any]) -> str:
        return str(
            paper.get("screening_result", {}).get("relevance_level", "")
        ).lower()

    def _get_lightweight_content(self, paper: Dict[str, Any]) -> tuple[str, str]:
        abstract_note = paper.get("abstract_note", "") or ""
        research_content = paper.get("research_content", "") or ""

        if abstract_note.strip():
            return "abstract_note", abstract_note

        return "research_content", research_content

    def _get_full_content(self, paper: Dict[str, Any]) -> tuple[str, str]:
        expert_obj = paper.get("expert_level_deep_analysis", {})

        if isinstance(expert_obj, dict):
            final_deep = expert_obj.get("final_deep_analysis", "") or ""
            if final_deep.strip():
                return "expert_level_deep_analysis", final_deep

        return "query_centered_content", paper.get("query_centered_content", "") or ""

    def _compact_record_for_mindmap(
        self,
        paper: Dict[str, Any],
        survey_mode: str,
    ) -> Dict[str, Any]:
        base = {
            "paper_id": paper.get("paper_id", ""),
            "title": paper.get("title", ""),
            "year": paper.get("year", ""),
            "file_name": paper.get("file_name", ""),
            "research_region": paper.get("research_region", ""),
            "relevance_level": self._relevance_level(paper),
        }

        if survey_mode == "lightweight":
            source, content = self._get_lightweight_content(paper)

        elif survey_mode == "middle":
            source = "query_centered_content"
            content = paper.get("query_centered_content", "") or ""

        elif survey_mode == "full":
            source, content = self._get_full_content(paper)

        else:
            raise ValueError(f"Unsupported survey_mode: {survey_mode}")

        base["record_source"] = source
        base["record_content"] = content

        return base

    def _compact_records_for_mindmap(
        self,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
    ) -> List[Dict[str, Any]]:
        paper_records = sorted(paper_records, key=self._safe_year)

        return [
            self._compact_record_for_mindmap(
                paper=paper,
                survey_mode=survey_mode,
            )
            for paper in paper_records
        ]

    def generate_mindmap(
        self,
        query: str,
        survey_text: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
    ) -> str:
        compact_records = self._compact_records_for_mindmap(
            paper_records=paper_records,
            survey_mode=survey_mode,
        )

        user_prompt = FINAL_MINDMAP_USER_PROMPT.format(
            query=query,
            survey_text=survey_text,
            paper_records_json=json.dumps(
                compact_records,
                ensure_ascii=False,
                indent=2,
            ),
        )

        return self._call_llm(
            system_prompt=FINAL_MINDMAP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def generate_and_save(
        self,
        query: str,
        survey_text: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
        output_path: str,
    ) -> str:
        mindmap_text = self.generate_mindmap(
            query=query,
            survey_text=survey_text,
            paper_records=paper_records,
            survey_mode=survey_mode,
        )

        write_text(output_path, mindmap_text)
        return mindmap_text