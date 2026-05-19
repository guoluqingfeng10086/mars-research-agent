# src/lit_agent/agents/survey_agent.py

import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config import (
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
)

from lit_agent.prompts.survey_prompts import (
    FINAL_SURVEY_SYSTEM_PROMPT,
    FINAL_SURVEY_USER_PROMPT,
)
from lit_agent.utils.io_utils import write_text


class SurveyAgent:
    """
    Generate final survey.txt.

    lightweight:
        LLM input uses abstract_note or research_content.

    middle:
        LLM input uses query_centered_content.

    full:
        medium papers use query_centered_content.
        high papers use:
            - expert_level_deep_analysis.final_deep_analysis as record_content
            - reasoning_trace for evidence-chain explanation.
    """

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        temperature: float = 0.2,
        max_prompt_tokens: int = 60000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_prompt_tokens = max_prompt_tokens

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

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count before final LLM call.

        Prefer tiktoken if available.
        If tiktoken is unavailable, use a conservative character-based estimate.
        """
        text = text or ""

        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except Exception:
            # Conservative fallback for mixed JSON/scientific English.
            # English is often about 4 chars/token, but JSON and mixed text may be denser.
            return max(1, int(len(text) / 3))

    def _safe_year(self, item: Dict[str, Any]) -> int:
        try:
            return int(float(item.get("year", 9999)))
        except Exception:
            return 9999

    def _year_str(self, item: Dict[str, Any]) -> str:
        year = item.get("year", "")

        try:
            return str(int(float(year)))
        except Exception:
            return str(year).strip()

    def _relevance_level(self, paper: Dict[str, Any]) -> str:
        return str(
            paper.get("screening_result", {}).get("relevance_level", "")
        ).lower()

    def _extract_first_author_lastname(self, authors: str) -> str:
        """
        Convert author string to first-author surname.

        Examples:
            "Fairén, Alberto G.; ..." -> "Fairén"
            "A. G. Fairén; ..." -> "Fairén"
        """
        authors = (authors or "").strip()

        if not authors:
            return ""

        first_author = authors.split(";")[0].strip()

        if not first_author:
            return ""

        if "," in first_author:
            lastname = first_author.split(",")[0].strip()
        else:
            parts = first_author.split()
            lastname = parts[-1].strip() if parts else ""

        return lastname

    def _citation(self, paper: Dict[str, Any]) -> str:
        """
        Build author-year citation.

        Example:
            (Fairén et al., 2003)
        """
        authors = paper.get("authors", "") or ""
        year = self._year_str(paper)
        lastname = self._extract_first_author_lastname(authors)

        if lastname and year:
            return f"({lastname} et al., {year})"

        if lastname:
            return f"({lastname} et al.)"

        if year:
            return f"({year})"

        return ""

    def _get_lightweight_content(self, paper: Dict[str, Any]) -> Tuple[str, str]:
        """
        Lightweight mode:
        use abstract_note first; if empty, use research_content.
        """
        abstract_note = paper.get("abstract_note", "") or ""
        research_content = paper.get("research_content", "") or ""

        if abstract_note.strip():
            return "abstract_note", abstract_note

        return "research_content", research_content

    def _get_reasoning_trace(
        self,
        expert_obj: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Build a compact public reasoning trace for survey writing.

        This is not hidden chain-of-thought.
        It only uses explicit fields produced by PaperDigestAgent.
        It is compatible with both current and older field names.
        """
        return {
            "round1": (
                expert_obj.get("round1_summary", "")
                or expert_obj.get("round1_relevance_framing", "")
                or expert_obj.get("round1_problem_framing", "")
                or ""
            ),
            "round2": (
                expert_obj.get("round2_deep_reasoning", "")
                or expert_obj.get("round2_evidence_reasoning", "")
                or ""
            ),
            "round3": (
                expert_obj.get("round3_focused_reasoning", "")
                or ""
            ),
        }

    def _get_full_content_and_trace(
        self,
        paper: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, str]]:
        """
        Full mode:
        - high papers usually contain expert_level_deep_analysis.
        - Use final_deep_analysis as record_content.
        - Add reasoning_trace if available.
        - If no expert analysis exists, fallback to query_centered_content.
        """
        expert_obj = paper.get("expert_level_deep_analysis", {})

        if isinstance(expert_obj, dict):
            final_deep = expert_obj.get("final_deep_analysis", "") or ""

            if final_deep.strip():
                reasoning_trace = self._get_reasoning_trace(expert_obj)

                # Remove empty trace only if all rounds are empty.
                if not any(v.strip() for v in reasoning_trace.values()):
                    reasoning_trace = {}

                return (
                    "expert_level_deep_analysis",
                    final_deep,
                    reasoning_trace,
                )

        return (
            "query_centered_content",
            paper.get("query_centered_content", "") or "",
            {},
        )

    def _compact_record_for_survey(
        self,
        paper: Dict[str, Any],
        survey_mode: str,
    ) -> Dict[str, Any]:
        """
        Compact one paper record before sending it to the final survey LLM.

        This prevents retrieval scores, embeddings, raw screening fields,
        and full PDF text from being sent to the final survey stage.
        """
        base = {
            "paper_id": paper.get("paper_id", ""),
            "title": paper.get("title", ""),
            "year": paper.get("year", ""),
            "journal": paper.get("journal", ""),
            "authors": paper.get("authors", ""),
            "citation": self._citation(paper),
            "file_name": paper.get("file_name", ""),
            "research_region": paper.get("research_region", ""),
            "relevance_level": self._relevance_level(paper),
        }

        if survey_mode == "lightweight":
            source, content = self._get_lightweight_content(paper)
            base["record_source"] = source
            base["record_content"] = content

        elif survey_mode == "middle":
            base["record_source"] = "query_centered_content"
            base["record_content"] = paper.get("query_centered_content", "") or ""

        elif survey_mode == "full":
            source, content, reasoning_trace = self._get_full_content_and_trace(paper)
            base["record_source"] = source
            base["record_content"] = content

            if reasoning_trace:
                base["reasoning_trace"] = reasoning_trace

        else:
            raise ValueError(f"Unsupported survey_mode: {survey_mode}")

        return base

    def _compact_records_for_survey(
        self,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
    ) -> List[Dict[str, Any]]:
        paper_records = sorted(paper_records, key=self._safe_year)

        return [
            self._compact_record_for_survey(
                paper=paper,
                survey_mode=survey_mode,
            )
            for paper in paper_records
        ]

    def generate_survey(
        self,
        query: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
    ) -> str:
        compact_records = self._compact_records_for_survey(
            paper_records=paper_records,
            survey_mode=survey_mode,
        )

        user_prompt = FINAL_SURVEY_USER_PROMPT.format(
            query=query,
            paper_records_json=json.dumps(
                compact_records,
                ensure_ascii=False,
                indent=2,
            ),
        )

        system_tokens = self._estimate_tokens(FINAL_SURVEY_SYSTEM_PROMPT)
        user_tokens = self._estimate_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        print(f"[SurveyAgent] Final survey compact records: {len(compact_records)}")
        print(f"[SurveyAgent] System prompt chars: {len(FINAL_SURVEY_SYSTEM_PROMPT)}")
        print(f"[SurveyAgent] User prompt chars: {len(user_prompt)}")
        print(f"[SurveyAgent] Estimated system prompt tokens: {system_tokens}")
        print(f"[SurveyAgent] Estimated user prompt tokens: {user_tokens}")
        print(f"[SurveyAgent] Estimated total input tokens: {total_tokens}")
        print(f"[SurveyAgent] Token budget: {self.max_prompt_tokens}")

        if total_tokens > self.max_prompt_tokens:
            print(
                f"[SurveyAgent] Warning: estimated input tokens exceed budget "
                f"({total_tokens} > {self.max_prompt_tokens}). "
                f"The request will still be sent without compression."
            )

        return self._call_llm(
            system_prompt=FINAL_SURVEY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def generate_and_save(
        self,
        query: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
        output_path: str,
    ) -> str:
        survey_text = self.generate_survey(
            query=query,
            paper_records=paper_records,
            survey_mode=survey_mode,
        )

        write_text(output_path, survey_text)
        return survey_text