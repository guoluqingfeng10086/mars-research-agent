# src/lit_agent/agents/paper_digest_agent.py
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from config import (
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
    PDF_CORPUS_DIR,
    MD_CORPUS_DIR,
    DEFAULT_MAX_SOURCE_CHARS,
)

from lit_agent.prompts.survey_prompts import (
    PAPER_QUERY_CONTENT_SYSTEM_PROMPT,
    PAPER_QUERY_CONTENT_USER_PROMPT,
    PAPER_DEEP_ANALYSIS_SYSTEM_PROMPT,
    PAPER_DEEP_ANALYSIS_ROUND1_PROMPT,
    PAPER_DEEP_ANALYSIS_ROUND2_PROMPT,
    PAPER_DEEP_ANALYSIS_ROUND3_PROMPT,
    PAPER_DEEP_ANALYSIS_FINAL_PROMPT,
)
from lit_agent.utils.io_utils import read_text, save_json
from lit_agent.utils.md_utils import resolve_md_path, extract_md_text
from lit_agent.utils.pdf_utils import resolve_pdf_path, extract_pdf_text


class PaperDigestAgent:
    """
    Build paper records for lightweight / middle / full modes.

    lightweight:
        no PDF reading, no LLM call.

    middle:
        high + medium papers -> query_centered_content.

    full:
        medium papers -> query_centered_content.
        high papers   -> expert_level_deep_analysis.
    """

    def __init__(
        self,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        pdf_corpus_dir: str = PDF_CORPUS_DIR,
        md_corpus_dir: str = MD_CORPUS_DIR,
        content_source: str = "pdf",
        temperature: float = 0.2,
        max_pdf_chars: Optional[int] = None,
        digest_cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.pdf_corpus_dir = Path(pdf_corpus_dir)
        self.md_corpus_dir = Path(md_corpus_dir)
        self.content_source = content_source
        self.max_pdf_chars = (
            max_pdf_chars
            if max_pdf_chars is not None
            else DEFAULT_MAX_SOURCE_CHARS
        )
        self.use_cache = use_cache
        self.digest_cache_dir = Path(digest_cache_dir) if digest_cache_dir else None

        if self.digest_cache_dir is not None:
            self.digest_cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _parse_json(self, response: str) -> Any:
        text = self._strip_json_markdown(response)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON while generating paper record.\n"
                f"Raw response:\n{response}"
            ) from exc

    def _query_hash(self, query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()[:8]

    def _safe_filename(self, text: str, max_len: int = 120) -> str:
        text = re.sub(r"[\\/:*?\"<>|]+", "_", str(text))
        text = re.sub(r"\s+", "_", text).strip("_")
        return text[:max_len] if text else "unknown_paper"

    def _safe_year(self, paper: Dict[str, Any]) -> int:
        try:
            return int(float(paper.get("year", 9999)))
        except Exception:
            return 9999

    def _relevance_level(self, paper: Dict[str, Any]) -> str:
        return str(
            paper.get("screening_result", {}).get("relevance_level", "")
        ).lower()

    def _cache_path(
        self,
        query: str,
        paper: Dict[str, Any],
        mode: str,
        analysis_type: str,
    ) -> Optional[Path]:
        if self.digest_cache_dir is None:
            return None

        query_hash = self._query_hash(query)

        year = str(paper.get("year", "unknown")).replace(".0", "")
        title = paper.get("title") or paper.get("file_name") or "unknown_paper"
        title = self._safe_filename(title, max_len=50)

        paper_key = (
            paper.get("paper_id", "")
            or paper.get("file_name", "")
            or title
        )
        paper_hash = hashlib.md5(str(paper_key).encode("utf-8")).hexdigest()[:8]

        file_name = f"{query_hash}_{self.content_source}_{mode}_{analysis_type}_{year}_{paper_hash}_{title}.json"

        cache_path = self.digest_cache_dir / file_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        return cache_path

    def _read_source_for_paper(self, paper: Dict[str, Any]) -> Tuple[str, Path, str]:
        file_name = paper.get("file_name", "")

        if self.content_source == "pdf":
            source_path = resolve_pdf_path(
                file_name=file_name,
                pdf_root=self.pdf_corpus_dir,
            )
            source_label = "pdf_full_text"

            if source_path is None:
                source_path = resolve_md_path(
                    file_name=file_name,
                    md_root=self.md_corpus_dir,
                )
                source_label = "md_full_text"

                if source_path is None:
                    raise FileNotFoundError(
                        f"Cannot find PDF or Markdown for file_name={file_name} "
                        f"under {self.pdf_corpus_dir} or {self.md_corpus_dir}"
                    )

                print(
                    f"[PaperDigestAgent] PDF not found; falling back to Markdown: {source_path}"
                )
                source_text = extract_md_text(source_path)
            else:
                print(f"[PaperDigestAgent] Reading PDF: {source_path}")
                source_text = extract_pdf_text(source_path)

        elif self.content_source == "md":
            source_path = resolve_md_path(
                file_name=file_name,
                md_root=self.md_corpus_dir,
            )
            source_label = "md_full_text"

            if source_path is None:
                source_path = resolve_pdf_path(
                    file_name=file_name,
                    pdf_root=self.pdf_corpus_dir,
                )
                source_label = "pdf_full_text"

                if source_path is None:
                    raise FileNotFoundError(
                        f"Cannot find Markdown or PDF for file_name={file_name} "
                        f"under {self.md_corpus_dir} or {self.pdf_corpus_dir}"
                    )

                print(
                    f"[PaperDigestAgent] Markdown not found; falling back to PDF: {source_path}"
                )
                source_text = extract_pdf_text(source_path)
            else:
                print(f"[PaperDigestAgent] Reading Markdown: {source_path}")
                source_text = extract_md_text(source_path)

        else:
            raise ValueError(f"Unsupported content_source: {self.content_source}")

        if not source_text:
            raise ValueError(f"No text extracted from source: {source_path}")

        if self.max_pdf_chars is not None and len(source_text) > self.max_pdf_chars:
            print(
                f"[PaperDigestAgent] Source text is long: {len(source_text)} chars. "
                f"Truncating to {self.max_pdf_chars} chars."
            )
            source_text = source_text[: self.max_pdf_chars]

        return source_label, source_path, source_text

    def _compact_paper_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        content_source, content_text = self._select_content(paper)

        return {
            "paper_id": paper.get("paper_id", ""),
            "title": paper.get("title", ""),
            "year": paper.get("year", ""),
            "journal": paper.get("journal", ""),
            "authors": paper.get("authors", ""),
            "file_name": paper.get("file_name", ""),
            "research_region": paper.get("research_region", ""),
            "content_source": content_source,
            "content_text": content_text,
            "screening_result": paper.get("screening_result", {}),
        }

    def _select_content(self, paper: Dict[str, Any]) -> Tuple[str, str]:
        """
        Use abstract_note first. If empty, use research_content.
        Only one content field is included in paper metadata.
        """
        abstract_note = paper.get("abstract_note", "") or ""
        research_content = paper.get("research_content", "") or ""

        if abstract_note.strip():
            return "abstract_note", abstract_note

        return "research_content", research_content

    def build_lightweight_records(
        self,
        selected_papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records = []

        for paper in sorted(selected_papers, key=self._safe_year):
            item = dict(paper)
            item["record_mode"] = "lightweight"
            item["record_source"] = "abstract_note"
            records.append(item)

        return records

    def build_query_centered_record(
        self,
        query: str,
        paper: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        cache_path = self._cache_path(
            query=query,
            paper=paper,
            mode=mode,
            analysis_type="query_content",
        )

        if self.use_cache and cache_path is not None and cache_path.exists():
            print(f"[PaperDigestAgent] Using cached query-centered record: {cache_path}")
            return json.loads(read_text(cache_path))

        source_label, source_path, source_text = self._read_source_for_paper(paper)

        paper_meta = self._compact_paper_metadata(paper)
        paper_meta["resolved_source_path"] = str(source_path)

        user_prompt = PAPER_QUERY_CONTENT_USER_PROMPT.format(
            query=query,
            paper_json=json.dumps(
                paper_meta,
                ensure_ascii=False,
                indent=2,
            ),
            pdf_text=source_text,
        )

        response = self._call_llm(
            system_prompt=PAPER_QUERY_CONTENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        parsed = self._parse_json(response)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Expected JSON object for query-centered content, "
                f"but got {type(parsed)}"
            )

        merged = dict(paper)
        merged["record_mode"] = mode
        merged["record_source"] = "query_centered_content"
        merged["digest_source_field"] = source_label
        merged["resolved_source_path"] = str(source_path)
        merged["query_centered_content"] = parsed.get("query_centered_content", "")

        if cache_path is not None:
            save_json(merged, cache_path)
            print(f"[PaperDigestAgent] Saved query-centered record: {cache_path}")

        return merged

    def build_deep_analysis_record(
        self,
        query: str,
        paper: Dict[str, Any],
    ) -> Dict[str, Any]:
        cache_path = self._cache_path(
            query=query,
            paper=paper,
            mode="full",
            analysis_type="deep_analysis",
        )

        if self.use_cache and cache_path is not None and cache_path.exists():
            print(f"[PaperDigestAgent] Using cached deep-analysis record: {cache_path}")
            return json.loads(read_text(cache_path))

        source_label, source_path, source_text = self._read_source_for_paper(paper)

        paper_meta = self._compact_paper_metadata(paper)
        paper_meta["resolved_source_path"] = str(source_path)

        paper_json = json.dumps(
            paper_meta,
            ensure_ascii=False,
            indent=2,
        )

        # Round 1: initial synthesis and follow-up questions
        round1_prompt = PAPER_DEEP_ANALYSIS_ROUND1_PROMPT.format(
            query=query,
            paper_json=paper_json,
            pdf_text=source_text,
        )

        round1_response = self._call_llm(
            system_prompt=PAPER_DEEP_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=round1_prompt,
        )

        round1_obj = self._parse_json(round1_response)

        if not isinstance(round1_obj, dict):
            raise ValueError(
                "Expected JSON object for deep analysis Round 1, "
                f"but got {type(round1_obj)}"
            )

        round1_json = json.dumps(
            round1_obj,
            ensure_ascii=False,
            indent=2,
        )

        # Round 2: answer follow-up questions and reason over evidence
        round2_prompt = PAPER_DEEP_ANALYSIS_ROUND2_PROMPT.format(
            query=query,
            paper_json=paper_json,
            round1_json=round1_json,
            pdf_text=source_text,
        )

        round2_response = self._call_llm(
            system_prompt=PAPER_DEEP_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=round2_prompt,
        )

        round2_obj = self._parse_json(round2_response)

        if not isinstance(round2_obj, dict):
            raise ValueError(
                "Expected JSON object for deep analysis Round 2, "
                f"but got {type(round2_obj)}"
            )

        # Enforce consistency between needs_round3 and round3_focus_questions.
        needs_round3 = bool(round2_obj.get("needs_round3", False))
        round3_focus_questions = round2_obj.get("round3_focus_questions", [])

        if not isinstance(round3_focus_questions, list):
            round3_focus_questions = []

        if needs_round3 and len(round3_focus_questions) == 0:
            needs_round3 = False
            round2_obj["needs_round3"] = False
            round2_obj["round3_focus_questions"] = []

        if not needs_round3:
            round2_obj["needs_round3"] = False
            round2_obj["round3_focus_questions"] = []

        round2_json = json.dumps(
            round2_obj,
            ensure_ascii=False,
            indent=2,
        )

        # Optional Round 3
        round3_obj = {}

        if needs_round3:
            round3_prompt = PAPER_DEEP_ANALYSIS_ROUND3_PROMPT.format(
                query=query,
                paper_json=paper_json,
                round1_json=round1_json,
                round2_json=round2_json,
                round3_focus_questions=json.dumps(
                    round3_focus_questions,
                    ensure_ascii=False,
                    indent=2,
                ),
                pdf_text=source_text,
            )

            round3_response = self._call_llm(
                system_prompt=PAPER_DEEP_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=round3_prompt,
            )

            round3_obj = self._parse_json(round3_response)

            if not isinstance(round3_obj, dict):
                raise ValueError(
                    "Expected JSON object for deep analysis Round 3, "
                    f"but got {type(round3_obj)}"
                )

        round3_json = json.dumps(
            round3_obj,
            ensure_ascii=False,
            indent=2,
        )

        # Final expert synthesis
        final_prompt = PAPER_DEEP_ANALYSIS_FINAL_PROMPT.format(
            query=query,
            paper_json=paper_json,
            round1_json=round1_json,
            round2_json=round2_json,
            round3_json=round3_json,
        )

        final_response = self._call_llm(
            system_prompt=PAPER_DEEP_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=final_prompt,
        )

        parsed_final = self._parse_json(final_response)

        if not isinstance(parsed_final, dict):
            raise ValueError(
                "Expected JSON object for final expert-level deep analysis, "
                f"but got {type(parsed_final)}"
            )

        expert_obj = parsed_final.get("expert_level_deep_analysis", {})

        if not isinstance(expert_obj, dict):
            expert_obj = {}

        expert_obj.setdefault("round1_summary", round1_obj.get("round1_summary", ""))
        expert_obj.setdefault("round2_deep_reasoning", round2_obj.get("round2_deep_reasoning", ""))
        expert_obj.setdefault("round3_focused_reasoning", round3_obj.get("round3_focused_reasoning", ""))
        expert_obj.setdefault("final_deep_analysis", "")

        merged = dict(paper)
        merged["record_mode"] = "full"
        merged["record_source"] = "expert_level_deep_analysis"
        merged["digest_source_field"] = source_label
        merged["resolved_source_path"] = str(source_path)
        merged["expert_level_deep_analysis"] = expert_obj

        # Fallback for downstream use.
        merged["query_centered_content"] = expert_obj.get("final_deep_analysis", "")

        if cache_path is not None:
            save_json(merged, cache_path)
            print(f"[PaperDigestAgent] Saved deep-analysis record: {cache_path}")

        return merged

    def build_middle_records(
        self,
        query: str,
        selected_papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected_papers = sorted(selected_papers, key=self._safe_year)
        records = []

        for idx, paper in enumerate(selected_papers, start=1):
            print("=" * 100)
            print(f"[PaperDigestAgent] Middle mode paper {idx}/{len(selected_papers)}")
            print(f"Relevance: {self._relevance_level(paper)}")
            print(f"Title: {paper.get('title', '')}")
            print(f"File: {paper.get('file_name', '')}")
            print("=" * 100)

            try:
                record = self.build_query_centered_record(
                    query=query,
                    paper=paper,
                    mode="middle",
                )
            except FileNotFoundError as exc:
                print(f"[PaperDigestAgent] Skip paper because source file was not found: {exc}")
                continue

            records.append(record)

        return sorted(records, key=self._safe_year)

    def build_full_records(
        self,
        query: str,
        selected_papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected_papers = sorted(selected_papers, key=self._safe_year)
        records = []

        for idx, paper in enumerate(selected_papers, start=1):
            relevance_level = self._relevance_level(paper)

            print("=" * 100)
            print(f"[PaperDigestAgent] Full mode paper {idx}/{len(selected_papers)}")
            print(f"Relevance: {relevance_level}")
            print(f"Title: {paper.get('title', '')}")
            print(f"File: {paper.get('file_name', '')}")
            print("=" * 100)

            try:
                if relevance_level == "high":
                    record = self.build_deep_analysis_record(
                        query=query,
                        paper=paper,
                    )
                else:
                    record = self.build_query_centered_record(
                        query=query,
                        paper=paper,
                        mode="full",
                    )
            except FileNotFoundError as exc:
                print(f"[PaperDigestAgent] Skip paper because source file was not found: {exc}")
                continue

            records.append(record)

        return sorted(records, key=self._safe_year)
