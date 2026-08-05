# src/lit_agent/agents/query_analyzer.py

import ast
import json
import re
from typing import Any, Dict, List

from openai import OpenAI

from config import (
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
    QUERY_LLM_TIMEOUT,
)

from lit_agent.utils.text_utils import clean_text
from lit_agent.prompts.Intent_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    QUERY_ANALYZER_USER_PROMPT,
)


class QueryPlan:
    def __init__(
        self,
        original_query: str,
        semantic_query: str,
        geological_concepts=None,
        target_regions=None,
        query_phrases=None,
        bm25_query: str = "",
    ):
        self.original_query = original_query
        self.semantic_query = semantic_query
        self.geological_concepts = list(geological_concepts or [])
        self.target_regions = list(target_regions or [])
        # BM25 only uses complete phrases.
        self.query_phrases = list(query_phrases or [])
        self.bm25_query = bm25_query


class QueryAnalyzer:
    """
    LLM-based query analyzer.

    It extracts actual geological / planetary science concepts and explicitly
    mentioned target regions from the query.

    The original query is used for vector search.
    The extracted complete phrases are used for BM25 search and phrase matching.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=QUERY_LLM_API_KEY,
            base_url=QUERY_LLM_BASE_URL,
            timeout=QUERY_LLM_TIMEOUT,
        )

    def analyze(self, query: str) -> QueryPlan:
        query = clean_text(query)

        extracted = self._extract_with_llm(query)

        geological_concepts = self._clean_list(
            extracted.get("geological_concepts", [])
        )

        target_regions = self._clean_list(
            extracted.get("target_regions", [])
        )

        query_phrases = self._deduplicate(
            geological_concepts + target_regions
        )

        bm25_query = " | ".join(query_phrases)

        return QueryPlan(
            original_query=query,
            semantic_query=query,
            geological_concepts=geological_concepts,
            target_regions=target_regions,
            query_phrases=query_phrases,
            bm25_query=bm25_query,
        )

    def _extract_with_llm(self, query: str) -> Dict[str, Any]:
        system_prompt = QUERY_ANALYZER_SYSTEM_PROMPT
        user_prompt = QUERY_ANALYZER_USER_PROMPT.format(query=query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=QUERY_LLM_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=QUERY_LLM_MODEL,
                messages=messages,
                temperature=0.0,
            )

        output = response.choices[0].message.content

        print("\n[LLM query analysis raw output]")
        print(output)
        print("[End LLM query analysis raw output]\n")

        return self._extract_json(output)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^```\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            obj_text = match.group(0).strip()

            try:
                return json.loads(obj_text)
            except json.JSONDecodeError:
                pass

            try:
                parsed = ast.literal_eval(obj_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raise ValueError(f"Failed to parse LLM output as JSON:\n{text}")

    def _clean_list(self, values) -> List[str]:
        if not isinstance(values, list):
            return []

        cleaned = []

        for value in values:
            value = clean_text(value)
            if value:
                cleaned.append(value)

        return self._deduplicate(cleaned)

    def _deduplicate(self, values: List[str]) -> List[str]:
        seen = set()
        out = []

        for value in values:
            value = clean_text(value)

            if not value:
                continue

            key = value.lower()

            if key not in seen:
                seen.add(key)
                out.append(value)

        return out
