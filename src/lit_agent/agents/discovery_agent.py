"""Discovery workflow over survey memory, literature, and geologic evidence."""

import hashlib
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
from lit_agent.agents.discovery_evidence_agent import DiscoveryEvidenceAgent
from lit_agent.agents.discovery_gap_agent import DiscoveryGapAgent
from lit_agent.agents.research_map_agent import ResearchMapAgent
from lit_agent.prompts.discovery_prompts import (
    DETECTION_CONFIDENCE_SYSTEM_PROMPT,
    DETECTION_CONFIDENCE_USER_PROMPT,
    DISCOVERY_SYSTEM_PROMPT,
    DISCOVERY_USER_PROMPT,
    HYPOTHESIS_EVALUATION_SYSTEM_PROMPT,
    HYPOTHESIS_EVALUATION_USER_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
)
from lit_agent.retrieval.geo_context_retriever import query_and_summarize_geo_context
from lit_agent.utils.coordinate_utils import extract_coordinate, tokenize_for_match
from lit_agent.utils.io_utils import load_json, read_text, save_json, write_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_EVIDENCE_SOURCES = {"survey", "local", "web", "geo"}


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
        research_map_agent: Optional[ResearchMapAgent] = None,
        gap_agent: Optional[DiscoveryGapAgent] = None,
        evidence_agent: Optional[DiscoveryEvidenceAgent] = None,
    ):
        self.output_root_dir = output_root_dir
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.client = None
        self.research_map_agent = research_map_agent
        self.gap_agent = gap_agent
        self.evidence_agent = evidence_agent

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
        evidence_sources: Optional[List[str]] = None,
        enable_gap_filling: bool = True,
        max_discovery_rounds: int = 2,
        max_gaps_per_round: int = 3,
        local_k: int = 5,
        web_k: int = 5,
    ) -> Dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("Scientific question cannot be empty.")
        if max_discovery_rounds < 0:
            raise ValueError("max_discovery_rounds must be >= 0.")
        if max_gaps_per_round < 1:
            raise ValueError("max_gaps_per_round must be >= 1.")

        sources = self._normalize_evidence_sources(evidence_sources)
        coordinate = extract_coordinate(question)
        coordinate_dict = coordinate.as_dict() if coordinate else None
        output_dir = self._build_discovery_output_dir(question)

        needs_survey = bool({"survey", "local", "web"} & set(sources))
        best: Optional[SurveyRunCandidate] = None
        created_new_survey = False
        survey_text = ""
        mindmap_text = ""

        if needs_survey:
            candidates = self._list_recent_survey_runs(limit=top_n_dirs)
            best = self._select_best_candidate(question, candidates)
            if (
                best is None
                or best.score < min_reuse_score
                or not self._has_survey_context(best.path)
            ):
                if not run_survey_if_missing:
                    raise RuntimeError(
                        "No reusable survey run found and survey generation is disabled."
                    )
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

        matched = best or SurveyRunCandidate(
            path=PROJECT_ROOT,
            name="no_survey_context",
            query="No survey context requested.",
            score=0.0,
        )

        research_map = self._load_or_build_research_map(
            question=question,
            matched_run=best,
            survey_text=survey_text,
            mindmap_text=mindmap_text,
        )
        save_json(research_map, output_dir / "initial_research_map.json")

        geo_context: Optional[Dict[str, Any]] = None
        geo_summary = ""
        acquired_evidence: List[Dict[str, Any]] = []
        if coordinate is not None and "geo" in sources:
            geo_context, geo_summary = query_and_summarize_geo_context(
                lat=coordinate.lat,
                lon=coordinate.lon_180,
            )
            if geo_summary.strip():
                acquired_evidence.append(
                    self._make_geo_context_evidence(coordinate_dict, geo_summary)
                )
                research_map = self._research_map_agent().update(
                    research_map, acquired_evidence
                )

        gap_sources = [source for source in sources if source in {"local", "web", "geo"}]
        gap_result = self._default_gap_result()
        stop_reason = "gap_filling_disabled"
        rounds: List[Dict[str, Any]] = []
        seen_evidence_ids = {
            item.get("evidence_id") for item in acquired_evidence if item.get("evidence_id")
        }

        for round_index in range(max_discovery_rounds + 1):
            try:
                question_subgraph = self._research_map_agent().build_question_subgraph(
                    research_map
                )
                gap_result = self._gap_agent().analyze(
                    question=question,
                    research_map=question_subgraph,
                    acquired_evidence=acquired_evidence,
                    allowed_sources=gap_sources,
                    coordinate=coordinate_dict,
                    max_gaps=max_gaps_per_round,
                )
            except Exception as exc:
                gap_result = self._default_gap_result(
                    reason=f"Completeness analysis failed: {type(exc).__name__}: {exc}"
                )
                stop_reason = "completeness_analysis_failed"
                round_record = {
                    "round": round_index,
                    "gap_result": gap_result,
                    "acquisition": {"evidence": [], "tasks": [], "errors": []},
                }
                rounds.append(round_record)
                self._save_round(output_dir, round_record)
                break

            round_record = {
                "round": round_index,
                "gap_result": gap_result,
                "acquisition": {"evidence": [], "tasks": [], "errors": []},
            }

            if gap_result.get("is_complete"):
                stop_reason = "sufficient_evidence"
                rounds.append(round_record)
                self._save_round(output_dir, round_record)
                break
            if not enable_gap_filling:
                stop_reason = "gap_filling_disabled"
                rounds.append(round_record)
                self._save_round(output_dir, round_record)
                break
            if round_index >= max_discovery_rounds:
                stop_reason = "max_rounds_reached"
                rounds.append(round_record)
                self._save_round(output_dir, round_record)
                break

            actionable_gaps = [
                gap
                for gap in gap_result.get("gaps", [])
                if gap.get("recommended_source") in gap_sources
                and gap.get("recommended_source") != "none"
            ]
            if not actionable_gaps:
                stop_reason = "no_actionable_gap"
                rounds.append(round_record)
                self._save_round(output_dir, round_record)
                break

            acquisition = self._evidence_agent().acquire(
                question=question,
                gaps=actionable_gaps,
                coordinate=coordinate_dict,
                allowed_sources=gap_sources,
                local_k=local_k,
                web_k=web_k,
                content_source=content_source,
                survey_mode=survey_mode,
            )
            new_evidence = []
            for item in acquisition.get("evidence", []):
                evidence_id = item.get("evidence_id")
                if not evidence_id or evidence_id in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(evidence_id)
                new_evidence.append(item)
            acquisition["evidence"] = new_evidence
            round_record["acquisition"] = acquisition
            rounds.append(round_record)
            self._save_round(output_dir, round_record)

            if not new_evidence:
                stop_reason = "no_new_evidence"
                break
            acquired_evidence.extend(new_evidence)
            research_map = self._research_map_agent().update(
                research_map=research_map,
                new_evidence=new_evidence,
            )
        else:
            stop_reason = "max_rounds_reached"

        save_json(research_map, output_dir / "final_research_map.json")
        save_json(rounds, output_dir / "discovery_rounds.json")
        save_json(gap_result, output_dir / "final_completeness.json")

        report = self._generate_report(
            question=question,
            matched_run=matched,
            survey_text=survey_text,
            mindmap_text=mindmap_text,
            geo_summary=geo_summary,
            research_map=research_map,
            acquired_evidence=acquired_evidence,
            gap_result=gap_result,
            stop_reason=stop_reason,
            prompt_mode=prompt_mode,
        )
        write_text(output_dir / "discovery_report.txt", report)
        if geo_summary:
            write_text(output_dir / "geo_context_summary.txt", geo_summary)

        result = {
            "question": question,
            "matched_run_dir": str(best.path) if best else None,
            "matched_run_name": best.name if best else None,
            "source_survey_stage_dir": str(best.path / "survey_stage") if best else None,
            "matched_query": best.query if best else None,
            "match_score": best.score if best else 0.0,
            "created_new_survey": created_new_survey,
            "coordinates": coordinate_dict,
            "used_geo_context": bool(geo_summary),
            "geo_context": geo_context,
            "prompt_mode": self._resolve_prompt_mode(question, prompt_mode),
            "evidence_sources": sources,
            "acquired_evidence_count": len(acquired_evidence),
            "discovery_round_count": len(rounds),
            "stop_reason": stop_reason,
            "completeness": gap_result,
            "discovery_report": report,
            "output_dir": str(output_dir),
        }
        save_json(result, output_dir / "discovery_result.json")
        return result

    def _research_map_agent(self) -> ResearchMapAgent:
        if self.research_map_agent is None:
            self.research_map_agent = ResearchMapAgent(temperature=0.0)
        return self.research_map_agent

    def _gap_agent(self) -> DiscoveryGapAgent:
        if self.gap_agent is None:
            self.gap_agent = DiscoveryGapAgent(temperature=0.0)
        return self.gap_agent

    def _evidence_agent(self) -> DiscoveryEvidenceAgent:
        if self.evidence_agent is None:
            self.evidence_agent = DiscoveryEvidenceAgent()
        return self.evidence_agent

    def _normalize_evidence_sources(
        self, evidence_sources: Optional[List[str]]
    ) -> List[str]:
        values = evidence_sources or ["survey", "local", "web", "geo"]
        normalized = []
        for value in values:
            source = str(value or "").strip().lower()
            if source not in VALID_EVIDENCE_SOURCES:
                raise ValueError(f"Unsupported evidence source: {source}")
            if source not in normalized:
                normalized.append(source)
        if not normalized:
            raise ValueError("At least one evidence source is required.")
        return normalized

    def _default_gap_result(self, reason: str = "Completeness was not evaluated.") -> Dict[str, Any]:
        return {
            "is_complete": False,
            "answerability": "insufficient",
            "reason": reason,
            "completeness": {
                "coverage": 0.0,
                "directness": 0.0,
                "source_quality": 0.0,
                "spatial_fit": 0.0,
                "consistency": 0.0,
                "alternative_coverage": 0.0,
            },
            "gaps": [],
        }

    def _make_geo_context_evidence(
        self, coordinate: Dict[str, Any], geo_summary: str
    ) -> Dict[str, Any]:
        source_id = f"{coordinate['lat']}|{coordinate['lon_180']}"
        digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
        return {
            "evidence_id": f"DGEO{digest}",
            "gap_id": "initial_local_context",
            "source_type": "geo_product",
            "source_id": source_id,
            "reading_level": "data_product_summary",
            "content": geo_summary,
            "location": coordinate,
            "limitations": [
                "Product availability and resolution differ across the queried location."
            ],
        }

    def _save_round(self, output_dir: Path, round_record: Dict[str, Any]) -> None:
        round_index = int(round_record.get("round", 0))
        save_json(
            round_record,
            output_dir / "rounds" / f"round_{round_index:03d}.json",
        )

    def _output_root(self) -> Path:
        root = Path(self.output_root_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return root

    def _build_discovery_output_dir(self, question: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        keyword = _sanitize_for_dir(question, max_len=80)
        dirname = f"{keyword}_{timestamp}" if keyword else timestamp
        path = PROJECT_ROOT / "discovery_stage" / dirname
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        config = load_json(run_dir / "run_config.json", default={}) or {}
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
                text += " " + read_text(mindmap_path)[:3000]
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
            score += self._specificity_boost(question_tokens, folder_tokens)
            scored.append(
                SurveyRunCandidate(
                    path=candidate.path,
                    name=candidate.name,
                    query=candidate.query,
                    score=round(min(score, 1.0), 4),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[0]

    def _specificity_boost(self, question_tokens, folder_tokens) -> float:
        """Give a small domain-neutral boost to shared specific terms."""
        shared = question_tokens & folder_tokens
        specific = [token for token in shared if len(token) >= 7 or any(ch.isdigit() for ch in token)]
        return min(0.24, 0.06 * len(specific))

    def _has_survey_context(self, run_dir: Path) -> bool:
        survey_dir = run_dir / "survey_stage"
        return (survey_dir / "survey.txt").exists() and (survey_dir / "MindMap.txt").exists()

    def _load_survey_context(self, run_dir: Path) -> Tuple[str, str]:
        survey_dir = run_dir / "survey_stage"
        survey_text = read_text(survey_dir / "survey.txt")
        mindmap_text = read_text(survey_dir / "MindMap.txt")
        if not survey_text.strip() or not mindmap_text.strip():
            raise RuntimeError(f"Missing survey.txt or MindMap.txt in {survey_dir}")
        return survey_text, mindmap_text

    def _load_paper_records(self, run_dir: Path) -> List[Dict[str, Any]]:
        survey_dir = run_dir / "survey_stage"
        for name in [
            "paper_records_full.json",
            "paper_records_middle.json",
            "paper_records_lightweight.json",
            "selected_papers.json",
        ]:
            data = load_json(survey_dir / name, default=None)
            if isinstance(data, list) and data:
                return data
        return []

    def _load_or_build_research_map(
        self,
        question: str,
        matched_run: Optional[SurveyRunCandidate],
        survey_text: str,
        mindmap_text: str,
    ) -> Dict[str, Any]:
        if matched_run is None:
            return self._research_map_agent().empty_map(question)
        path = matched_run.path / "survey_stage" / "research_map.json"
        existing = load_json(path, default=None)
        if isinstance(existing, dict):
            return existing
        research_map = self._research_map_agent().generate(
            question=question,
            survey_text=survey_text,
            mindmap_text=mindmap_text,
            paper_records=self._load_paper_records(matched_run.path),
        )
        save_json(research_map, path)
        return research_map

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
        research_map: Dict[str, Any],
        acquired_evidence: List[Dict[str, Any]],
        gap_result: Dict[str, Any],
        stop_reason: str,
        prompt_mode: str = "auto",
    ) -> str:
        resolved_mode = self._resolve_prompt_mode(question, prompt_mode)
        if resolved_mode in {"detection_confidence", "mineral_confidence"}:
            system_prompt = DETECTION_CONFIDENCE_SYSTEM_PROMPT
            user_template = DETECTION_CONFIDENCE_USER_PROMPT
        elif resolved_mode == "hypothesis_evaluation":
            system_prompt = HYPOTHESIS_EVALUATION_SYSTEM_PROMPT
            user_template = HYPOTHESIS_EVALUATION_USER_PROMPT
        else:
            system_prompt = DISCOVERY_SYSTEM_PROMPT
            user_template = DISCOVERY_USER_PROMPT

        user_prompt = user_template.format(
            question=question,
            matched_run=f"{matched_run.name} (score={matched_run.score})",
            research_map_json=self._trim_json(research_map, 50000),
            survey_text=self._trim(survey_text, 45000) or "None.",
            mindmap_text=self._trim(mindmap_text, 30000) or "None.",
            geo_context=geo_summary or "None.",
            acquired_evidence_json=self._trim_json(acquired_evidence, 40000),
            completeness_json=json.dumps(gap_result, ensure_ascii=False, indent=2),
            stop_reason=stop_reason,
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
        mode = str(prompt_mode or "auto").strip().lower()
        if mode in {
            "generic",
            "detection_confidence",
            "hypothesis_evaluation",
            "mineral_confidence",
        }:
            return mode
        if self._is_detection_confidence_question(question):
            return "detection_confidence"
        if self._is_hypothesis_question(question):
            return "hypothesis_evaluation"
        return "generic"

    def _is_detection_confidence_question(self, question: str) -> bool:
        text = str(question or "").lower()
        assessment_terms = [
            "confidence",
            "credible",
            "credibility",
            "reliable",
            "reliability",
            "classify",
            "level",
            "可信",
            "置信",
            "可靠",
            "分级",
        ]
        detection_terms = [
            "detection",
            "detected",
            "detect",
            "observation",
            "observed",
            "signal",
            "feature",
            "发现",
            "探测",
            "观测",
            "信号",
        ]
        return any(term in text for term in assessment_terms) and any(
            term in text for term in detection_terms
        )

    def _is_hypothesis_question(self, question: str) -> bool:
        text = str(question or "").lower()
        terms = [
            "hypothesis",
            "evidence for",
            "support the",
            "whether",
            "did an",
            "does the evidence",
            "假说",
            "证据支持",
            "是否存在",
            "是否支持",
        ]
        return any(term in text for term in terms)

    def _client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for discovery LLM calls."
            ) from exc
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self.client

    def _trim(self, text: str, max_chars: int) -> str:
        text = str(text or "")
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Truncated for prompt budget.]"

    def _trim_json(self, value: Any, max_chars: int) -> str:
        return self._trim(json.dumps(value, ensure_ascii=False, indent=2), max_chars)


def _sanitize_for_dir(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', " ", str(text or ""))
    text = re.sub(r"\s+", "_", text).strip("_")
    text = re.sub(r"[^A-Za-z0-9_\-.\u4e00-\u9fff]+", "", text)
    return text[:max_len]
