"""Discovery agent that reuses survey outputs and optional geologic products."""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    DEFAULT_TOP_K,
    OUTPUT_ROOT_DIR,
    QUERY_LLM_API_KEY,
    QUERY_LLM_BASE_URL,
    QUERY_LLM_MODEL,
    QUERY_LLM_TIMEOUT,
)
from lit_agent.prompts.discovery_prompts import (
    DISCOVERY_SYSTEM_PROMPT,
    DISCOVERY_USER_PROMPT,
    MINERAL_CONFIDENCE_SYSTEM_PROMPT,
    MINERAL_CONFIDENCE_USER_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
)
from lit_agent.retrieval.geo_context_retriever import query_and_summarize_geo_context
from lit_agent.utils.coordinate_utils import extract_coordinate, tokenize_for_match


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SurveyRunCandidate:
    def __init__(self, path: Path, name: str, query: str, score: float):
        self.path = path
        self.name = name
        self.query = query
        self.score = score


class DiscoveryAgent:
    def __init__(
        self,
        output_root_dir: str = OUTPUT_ROOT_DIR,
        model: str = QUERY_LLM_MODEL,
        api_key: str = QUERY_LLM_API_KEY,
        base_url: str = QUERY_LLM_BASE_URL,
        timeout: int = QUERY_LLM_TIMEOUT,
        temperature: float = 0.2,
    ):
        self.output_root_dir = output_root_dir
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.client = None

    @classmethod
    def from_config(cls) -> "DiscoveryAgent":
        return cls()

    def run(
        self,
        question: str,
        top_n_dirs: int = 10,
        min_reuse_score: float = 0.35,
        top_m: int = DEFAULT_TOP_K,
        survey_mode: str = "full",
        content_source: str = "md",
        run_survey_if_missing: bool = True,
        prompt_mode: str = "auto",
    ) -> Dict[str, Any]:
        question = question.strip()
        coordinate = extract_coordinate(question)

        candidates = self._list_recent_survey_runs(limit=top_n_dirs)
        best = self._select_best_candidate(question, candidates)

        created_new_survey = False
        if (
            best is None
            or best.score < min_reuse_score
            or not self._has_survey_context(best.path)
        ):
            if not run_survey_if_missing:
                raise RuntimeError("No reusable survey run found and survey generation is disabled.")
            rewritten_query = self._rewrite_query(question)
            run_dir = self._run_survey(
                query=rewritten_query,
                top_m=top_m,
                survey_mode=survey_mode,
                content_source=content_source,
            )
            best = SurveyRunCandidate(
                path=run_dir,
                name=run_dir.name,
                query=rewritten_query,
                score=1.0,
            )
            created_new_survey = True

        survey_text, mindmap_text = self._load_survey_context(best.path)

        geo_context = None
        geo_summary = ""
        if coordinate is not None:
            geo_context, geo_summary = query_and_summarize_geo_context(
                lat=coordinate.lat,
                lon=coordinate.lon_180,
            )

        report = self._generate_report(
            question=question,
            matched_run=best,
            survey_text=survey_text,
            mindmap_text=mindmap_text,
            geo_summary=geo_summary,
            prompt_mode=prompt_mode,
        )

        output_dir = self._build_discovery_output_dir(question)
        _write_text(output_dir / "discovery_report.txt", report)
        if geo_summary:
            _write_text(output_dir / "geo_context_summary.txt", geo_summary)

        result = {
            "question": question,
            "matched_run_dir": str(best.path),
            "matched_run_name": best.name,
            "source_survey_stage_dir": str(best.path / "survey_stage"),
            "matched_query": best.query,
            "match_score": best.score,
            "created_new_survey": created_new_survey,
            "coordinates": coordinate.as_dict() if coordinate else None,
            "used_geo_context": bool(geo_summary),
            "geo_context": geo_context,
            "prompt_mode": self._resolve_prompt_mode(question, prompt_mode),
            "discovery_report": report,
            "output_dir": str(output_dir),
        }
        _save_json(result, output_dir / "discovery_result.json")
        return result

    def _output_root(self) -> Path:
        root = Path(self.output_root_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return root

    def _build_discovery_output_dir(self, question: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        keyword = _sanitize_for_dir(question, max_len=80)
        if keyword:
            dirname = f"{keyword}_{timestamp}"
        else:
            dirname = timestamp
        return _ensure_dir(PROJECT_ROOT / "discovery_stage" / dirname)

    def _list_recent_survey_runs(self, limit: int) -> List[SurveyRunCandidate]:
        root = self._output_root()
        if not root.exists():
            return []

        dirs = [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "survey_stage").exists()
        ]
        dirs = sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]

        return [
            SurveyRunCandidate(
                path=path,
                name=path.name,
                query=self._read_run_query(path),
                score=0.0,
            )
            for path in dirs
        ]

    def _read_run_query(self, run_dir: Path) -> str:
        config = _load_json(run_dir / "run_config.json", default={}) or {}
        return str(config.get("query", "") or "")

    def _select_best_candidate(
        self,
        question: str,
        candidates: List[SurveyRunCandidate],
    ) -> Optional[SurveyRunCandidate]:
        if not candidates:
            return None

        question_tokens = tokenize_for_match(question)
        scored = []
        for candidate in candidates:
            folder_text = " ".join([candidate.name, candidate.query])
            text = folder_text
            mindmap_path = candidate.path / "survey_stage" / "MindMap.txt"
            if mindmap_path.exists():
                text += " " + _read_text(mindmap_path)[:3000]
            folder_tokens = tokenize_for_match(folder_text)
            tokens = tokenize_for_match(text)
            folder_overlap = len(question_tokens & folder_tokens)
            overlap = len(question_tokens & tokens)
            union = max(1, len(question_tokens | tokens))
            folder_score = folder_overlap / max(1, len(question_tokens))
            context_score = overlap / union
            score = 0.75 * folder_score + 0.25 * context_score
            if question_tokens:
                score += 0.05 * overlap / len(question_tokens)
            score += self._science_key_boost(question_tokens, folder_tokens)
            scored.append(
                SurveyRunCandidate(
                    path=candidate.path,
                    name=candidate.name,
                    query=candidate.query,
                    score=round(min(score, 1.0), 4),
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[0]

    def _science_key_boost(self, question_tokens, folder_tokens) -> float:
        boost = 0.0

        regions = [
            {"utopia", "planitia"},
            {"chryse", "planitia"},
            {"northern", "lowlands"},
        ]
        minerals = [
            {"fe", "smectite"},
            {"mg", "smectite"},
            {"jarosite"},
            {"h2o", "ice"},
            {"hydrated", "minerals"},
        ]
        features = [
            {"paleolake"},
            {"crater"},
            {"valley"},
            {"mud", "volcanism"},
            {"allophane"},
            {"sulfate"},
            {"olivine"},
        ]

        for key in regions:
            if key <= question_tokens and key <= folder_tokens:
                boost += 0.28
        for key in minerals:
            if key <= question_tokens and key <= folder_tokens:
                boost += 0.35
        for key in features:
            if key <= question_tokens and key <= folder_tokens:
                boost += 0.18

        return boost

    def _has_survey_context(self, run_dir: Path) -> bool:
        survey_dir = run_dir / "survey_stage"
        return (
            (survey_dir / "survey.txt").exists()
            and (survey_dir / "MindMap.txt").exists()
        )

    def _load_survey_context(self, run_dir: Path) -> Tuple[str, str]:
        survey_dir = run_dir / "survey_stage"
        survey_text = _read_text(survey_dir / "survey.txt")
        mindmap_text = _read_text(survey_dir / "MindMap.txt")
        if not survey_text.strip() or not mindmap_text.strip():
            raise RuntimeError(f"Missing survey.txt or MindMap.txt in {survey_dir}")
        return survey_text, mindmap_text

    def _rewrite_query(self, question: str) -> str:
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": QUERY_REWRITE_USER_PROMPT.format(question=question),
                    },
                ],
                temperature=0.0,
            )
            rewritten = response.choices[0].message.content.strip()
            rewritten = re.sub(r"^```.*?\n|\n```$", "", rewritten).strip()
            return rewritten or question
        except Exception:
            return question

    def _run_survey(
        self,
        query: str,
        top_m: int,
        survey_mode: str,
        content_source: str,
    ) -> Path:
        output_root = self._output_root()
        output_root.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_survey_with_supplement.py"),
            "--query",
            query,
            "--top_m",
            str(top_m),
            "--survey_mode",
            survey_mode,
            "--content_source",
            content_source,
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        after = [path for path in output_root.iterdir() if path.is_dir()]
        new_dirs = [path for path in after if path.resolve() not in before]
        if new_dirs:
            return sorted(new_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        latest = sorted(after, key=lambda path: path.stat().st_mtime, reverse=True)
        if latest:
            return latest[0]
        raise RuntimeError("Survey script finished but no output directory was found.")

    def _generate_report(
        self,
        question: str,
        matched_run: SurveyRunCandidate,
        survey_text: str,
        mindmap_text: str,
        geo_summary: str,
        prompt_mode: str = "auto",
    ) -> str:
        resolved_mode = self._resolve_prompt_mode(question, prompt_mode)
        if resolved_mode == "mineral_confidence":
            system_prompt = MINERAL_CONFIDENCE_SYSTEM_PROMPT
            user_template = MINERAL_CONFIDENCE_USER_PROMPT
        else:
            system_prompt = DISCOVERY_SYSTEM_PROMPT
            user_template = DISCOVERY_USER_PROMPT

        user_prompt = user_template.format(
            question=question,
            matched_run=f"{matched_run.name} (score={matched_run.score})",
            mindmap_text=self._trim(mindmap_text, 30000),
            survey_text=self._trim(survey_text, 45000),
            geo_context=geo_summary or "None.",
        )
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _resolve_prompt_mode(self, question: str, prompt_mode: str = "auto") -> str:
        mode = (prompt_mode or "auto").strip().lower()
        if mode in {"generic", "mineral_confidence"}:
            return mode
        return "mineral_confidence" if self._is_mineral_confidence_question(question) else "generic"

    def _is_mineral_confidence_question(self, question: str) -> bool:
        text = str(question or "").lower()
        confidence_terms = [
            "confidence",
            "level",
            "classify",
            "reliable",
            "low-confidence",
            "置信度",
            "分级",
            "等级",
        ]
        detection_terms = [
            "detection",
            "detect",
            "mineral",
            "volatile",
            "ice",
            "h2o",
            "smectite",
            "jarosite",
            "sulfate",
            "olivine",
            "hematite",
            "探测",
            "矿物",
        ]
        return any(term in text for term in confidence_terms) and any(
            term in text for term in detection_terms
        )

    def _client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for discovery LLM calls. "
                "Install it in the Python environment used to run this script."
            ) from exc
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self.client

    def _trim(self, text: str, max_chars: int) -> str:
        text = text or ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Truncated for prompt budget.]"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(orient="records"))
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, ensure_ascii=False, indent=2)


def _sanitize_for_dir(text: str, max_len: int = 80) -> str:
    text = str(text or "").strip()
    text = re.sub(r'[\\/:*?"<>|]+', " ", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "", text)
    text = text.strip("_")
    return text[:max_len]
