# src/lit_agent/agents/relevance_agent.py

import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config import (
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
)

from lit_agent.prompts.survey_prompts import (
    RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_USER_PROMPT,
)
from lit_agent.utils.io_utils import chunk_list


class RelevanceAgent:
    """
    Screen retrieved papers using only:
        row_id, paper_id, research_region, abstract_note

    The LLM returns only:
        row_id, relevance_level, is_selected, reason

    The program preserves all original paper metadata.
    """

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        temperature: float = 0.0,
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

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        text = self._strip_json_markdown(response)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON during relevance screening.\n"
                f"Raw response:\n{response}"
            ) from exc

        if isinstance(data, dict) and "results" in data:
            data = data["results"]

        if not isinstance(data, list):
            raise ValueError(
                "Expected a JSON list during relevance screening, "
                f"but got: {type(data)}"
            )

        return data
    
    def _select_content(self, paper: Dict[str, Any]) -> Tuple[str, str]:
        """
        Use abstract_note first. If empty, use research_content.
        Only one content field is sent to the LLM.
        """
        abstract_note = paper.get("abstract_note", "") or ""
        research_content = paper.get("research_content", "") or ""

        if abstract_note.strip():
            return "abstract_note", abstract_note

        return "research_content", research_content

    def _compact_candidate(
        self,
        row_id: int,
        paper: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Only send:
            row_id, paper_id, research_region, content_source, content_text

        content_text = abstract_note if available, otherwise research_content.
        """
        content_source, content_text = self._select_content(paper)

        return {
            "row_id": row_id,
            "paper_id": paper.get("paper_id", ""),
            "research_region": paper.get("research_region", ""),
            "content_source": content_source,
            "content_text": content_text,
        }

    def screen_batch(
        self,
        query: str,
        papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates = [
            self._compact_candidate(row_id=i, paper=paper)
            for i, paper in enumerate(papers)
        ]

        user_prompt = RELEVANCE_USER_PROMPT.format(
            query=query,
            papers_json=json.dumps(
                candidates,
                ensure_ascii=False,
                indent=2,
            ),
        )

        response = self._call_llm(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return self._parse_json_response(response)

    def screen_top_m(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        batch_size: int = 5,
    ) -> List[Dict[str, Any]]:
        selected_papers = []

        for batch_index, batch in enumerate(chunk_list(papers, batch_size), start=1):
            print(f"[RelevanceAgent] Screening batch {batch_index}, papers={len(batch)}")

            screening_results = self.screen_batch(
                query=query,
                papers=batch,
            )

            result_by_row_id = {}

            for item in screening_results:
                try:
                    row_id = int(item.get("row_id"))
                    result_by_row_id[row_id] = item
                except Exception:
                    continue

            for row_id, paper in enumerate(batch):
                screening_result = result_by_row_id.get(row_id)

                if screening_result is None:
                    screening_result = {
                        "row_id": row_id,
                        "relevance_level": "unknown",
                        "is_selected": False,
                        "reason": "No valid screening result was returned for this paper.",
                    }

                relevance_level = str(
                    screening_result.get("relevance_level", "unknown")
                ).lower()

                is_selected = bool(screening_result.get("is_selected"))

                # Program-side safety check:
                # only high and medium papers are retained.
                if relevance_level not in ["high", "medium"]:
                    is_selected = False

                paper_with_screening = dict(paper)
                paper_with_screening["screening_result"] = {
                    "row_id": row_id,
                    "relevance_level": relevance_level,
                    "is_selected": is_selected,
                    "reason": screening_result.get("reason", ""),
                }

                if is_selected:
                    selected_papers.append(paper_with_screening)

        return selected_papers
