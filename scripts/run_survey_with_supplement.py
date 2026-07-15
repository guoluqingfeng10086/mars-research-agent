# scripts/run_survey_with_supplement.py
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer project source code.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from config import DEFAULT_TOP_K
except ImportError:
    DEFAULT_TOP_K = 30

try:
    from config import OUTPUT_ROOT_DIR
except ImportError:
    OUTPUT_ROOT_DIR = "outputs"

try:
    from config import MD_CORPUS_DIR
except ImportError:
    MD_CORPUS_DIR = r"G:\MMCorpus2000v1_md"

try:
    from config import (
        ENABLE_SUPPLEMENT_ROUND,
        SUPPLEMENT_MODE,
        SUPPLEMENT_LOCAL_K,
        SUPPLEMENT_WEB_K,
        SUPPLEMENT_MAX_GAPS,
        SUPPLEMENT_MAX_WEB_QUERIES_PER_GAP,
        SUPPLEMENT_WEB_ENGINES,
        SUPPLEMENT_FROM_YEAR,
        SUPPLEMENT_WEB_TIMEOUT,
        SUPPLEMENT_ADS_API_TOKEN,
        SUPPLEMENT_ENABLE_CANDIDATE_SCREENING,
        SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS,
    )
except ImportError:
    ENABLE_SUPPLEMENT_ROUND = True
    SUPPLEMENT_MODE = "auto"
    SUPPLEMENT_LOCAL_K = 5
    SUPPLEMENT_WEB_K = 5
    SUPPLEMENT_MAX_GAPS = 2
    SUPPLEMENT_MAX_WEB_QUERIES_PER_GAP = 2
    SUPPLEMENT_WEB_ENGINES = ["ads", "crossref"]
    SUPPLEMENT_FROM_YEAR = None
    SUPPLEMENT_WEB_TIMEOUT = 30
    SUPPLEMENT_ADS_API_TOKEN = ""
    SUPPLEMENT_ENABLE_CANDIDATE_SCREENING = True
    SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS = ["high", "medium"]

from lit_agent.agents.mindmap_agent import MindMapAgent
from lit_agent.agents.paper_digest_agent import PaperDigestAgent
from lit_agent.agents.relevance_agent import RelevanceAgent
from lit_agent.agents.retrieval_agent import RetrievalAgent
from lit_agent.agents.supplement_agent import SupplementAgent
from lit_agent.agents.survey_agent import SurveyAgent
from lit_agent.prompts.survey_prompts import (
    FINAL_SURVEY_SYSTEM_PROMPT,
    FINAL_SURVEY_USER_PROMPT,
)
from lit_agent.retrieval.web_literature_searcher import WebLiteratureSearcher
from lit_agent.utils.io_utils import ensure_dir, save_json, write_text

from run_survey_stage import (
    build_run_output_dirs,
    extract_run_keyword_from_query_plan,
    input_content_source,
    input_query,
    input_survey_mode,
    input_top_m,
    safe_year,
    save_run_config,
)


class SupplementedSurveyAgent(SurveyAgent):
    """
    SurveyAgent variant used only by this experimental entrypoint.

    The base SurveyAgent class and the original run_survey_stage.py flow remain
    unchanged.
    """

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        print("[SupplementedSurveyAgent] Calling final survey LLM in streaming mode...")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            stream=True,
        )

        chunks = []
        chunk_count = 0

        for event in stream:
            if not getattr(event, "choices", None):
                continue

            delta = event.choices[0].delta
            content = getattr(delta, "content", None)

            if not content:
                continue

            chunks.append(content)
            chunk_count += 1

            if chunk_count % 50 == 0:
                print(
                    f"[SupplementedSurveyAgent] Received streaming chunks: "
                    f"{chunk_count}"
                )

        text = "".join(chunks).strip()

        if not text:
            raise RuntimeError(
                "Final survey LLM stream finished without returning any text."
            )

        print(
            f"[SupplementedSurveyAgent] Final survey LLM returned "
            f"{len(text)} chars."
        )

        return text

    def generate_survey(
        self,
        query: str,
        paper_records: List[Dict[str, Any]],
        survey_mode: str,
        supplement_report: Optional[Dict[str, Any]] = None,
        supplement_agent: Optional[SupplementAgent] = None,
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

        if supplement_report and supplement_agent is not None:
            user_prompt += supplement_agent.format_report_for_survey(
                supplement_report
            )

        system_tokens = self._estimate_tokens(FINAL_SURVEY_SYSTEM_PROMPT)
        user_tokens = self._estimate_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        print(
            f"[SupplementedSurveyAgent] Final survey compact records: "
            f"{len(compact_records)}"
        )
        print(f"[SupplementedSurveyAgent] User prompt chars: {len(user_prompt)}")
        print(f"[SupplementedSurveyAgent] Estimated total input tokens: {total_tokens}")
        print(f"[SupplementedSurveyAgent] Token budget: {self.max_prompt_tokens}")

        if total_tokens > self.max_prompt_tokens:
            print(
                f"[SupplementedSurveyAgent] Warning: estimated input tokens exceed "
                f"budget ({total_tokens} > {self.max_prompt_tokens}). "
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
        supplement_report: Optional[Dict[str, Any]] = None,
        supplement_agent: Optional[SupplementAgent] = None,
    ) -> str:
        survey_text = self.generate_survey(
            query=query,
            paper_records=paper_records,
            survey_mode=survey_mode,
            supplement_report=supplement_report,
            supplement_agent=supplement_agent,
        )
        write_text(output_path, survey_text)
        return survey_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run literature survey stage with optional supplement round."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Scientific question. If not provided, input interactively.",
    )
    parser.add_argument(
        "--top_m",
        type=int,
        default=None,
        help="Number of top retrieved papers. If not provided, input interactively.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for relevance screening.",
    )
    parser.add_argument(
        "--survey_mode",
        type=str,
        choices=["lightweight", "middle", "full"],
        default=None,
        help="Survey mode: lightweight, middle, or full. If not provided, input interactively.",
    )
    parser.add_argument(
        "--content_source",
        type=str,
        choices=["pdf", "md"],
        default=None,
        help="Content source for middle/full mode. If not provided, input interactively.",
    )
    parser.add_argument(
        "--md_corpus_dir",
        type=str,
        default=MD_CORPUS_DIR,
        help="Markdown corpus root directory.",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=OUTPUT_ROOT_DIR,
        help="Root output directory.",
    )
    parser.add_argument(
        "--max_pdf_chars",
        type=int,
        default=None,
        help="Maximum characters read from each PDF in middle/full mode.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable paper record cache.",
    )

    return parser.parse_args()


def _paper_key(paper: Dict[str, Any]) -> str:
    return str(
        paper.get("paper_id")
        or paper.get("doi")
        or paper.get("title")
        or paper.get("file_name")
        or ""
    ).lower().strip()


def _unique_texts(values: List[str]) -> List[str]:
    seen = set()
    out = []

    for value in values:
        value = str(value or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)

    return out


def collect_supplement_queries(
    supplement_plan: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    local_queries = []
    web_queries = []

    for gap in supplement_plan.get("gaps", []):
        if not isinstance(gap, dict):
            continue
        local_queries.append(gap.get("local_query", ""))
        web_queries.extend(gap.get("web_queries", []) or [])

    return _unique_texts(local_queries), _unique_texts(web_queries)


def retrieve_local_supplement_candidates(
    retrieval_agent: RetrievalAgent,
    local_queries: List[str],
    top_m: int,
    local_k: int,
    core_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if local_k <= 0 or not local_queries:
        return []

    seen = {_paper_key(paper) for paper in core_papers if _paper_key(paper)}
    candidates = []
    per_query_k = max(1, math.ceil(local_k / max(1, len(local_queries))))

    for local_query in local_queries:
        retrieve_k = max(top_m + per_query_k, per_query_k)
        results = retrieval_agent.retrieve(
            query=local_query,
            top_k=retrieve_k,
            normalize=True,
        )

        added_for_query = 0
        for paper in results:
            key = _paper_key(paper)
            if not key or key in seen:
                continue

            item = dict(paper)
            item["supplement_query"] = local_query
            candidates.append(item)
            seen.add(key)
            added_for_query += 1

            if added_for_query >= per_query_k or len(candidates) >= local_k:
                break

        if len(candidates) >= local_k:
            break

    return candidates[:local_k]


def run_supplement_round(
    query: str,
    survey_mode: str,
    top_m: int,
    paper_records: List[Dict[str, Any]],
    retrieval_agent: RetrievalAgent,
    survey_output_dir: Path,
) -> tuple[Optional[Dict[str, Any]], Optional[SupplementAgent]]:
    supplement_agent = SupplementAgent(temperature=0.0)

    supplement_config = {
        "enable_supplement_round": ENABLE_SUPPLEMENT_ROUND,
        "supplement_mode": SUPPLEMENT_MODE,
        "supplement_local_k": SUPPLEMENT_LOCAL_K,
        "supplement_web_k": SUPPLEMENT_WEB_K,
        "supplement_max_gaps": SUPPLEMENT_MAX_GAPS,
        "supplement_max_web_queries_per_gap": SUPPLEMENT_MAX_WEB_QUERIES_PER_GAP,
        "supplement_web_engines": SUPPLEMENT_WEB_ENGINES,
        "supplement_from_year": SUPPLEMENT_FROM_YEAR,
        "supplement_web_timeout": SUPPLEMENT_WEB_TIMEOUT,
        "supplement_enable_candidate_screening": SUPPLEMENT_ENABLE_CANDIDATE_SCREENING,
        "supplement_accept_relevance_levels": SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS,
    }
    save_json(supplement_config, survey_output_dir / "supplement_config.json")

    if not ENABLE_SUPPLEMENT_ROUND:
        report = {
            "supplement_used": False,
            "overall_effect": "none",
            "evidence_level": "abstract_or_summary_level_supplement",
            "summary": "Supplement round disabled by config.",
            "accepted_supplementary_papers": [],
            "integration_analysis": [],
            "follow_up_candidates": [],
        }
        save_json(report, survey_output_dir / "supplement_report.json")
        write_text(
            survey_output_dir / "supplementary_evidence_report.md",
            supplement_agent.format_report_as_markdown(report),
        )
        return None, supplement_agent

    supplement_plan = supplement_agent.plan_supplement(
        query=query,
        paper_records=paper_records,
        survey_mode=survey_mode,
        mode=SUPPLEMENT_MODE,
        max_gaps=SUPPLEMENT_MAX_GAPS,
        max_web_queries_per_gap=SUPPLEMENT_MAX_WEB_QUERIES_PER_GAP,
    )
    save_json(supplement_plan, survey_output_dir / "supplement_plan.json")

    if not supplement_plan.get("should_supplement"):
        report = {
            "supplement_used": False,
            "overall_effect": "none",
            "evidence_level": "abstract_or_summary_level_supplement",
            "summary": supplement_plan.get("reason", "No supplement needed."),
            "accepted_supplementary_papers": [],
            "integration_analysis": [],
            "follow_up_candidates": [],
            "supplement_plan": supplement_plan,
        }
        save_json(report, survey_output_dir / "supplement_report.json")
        write_text(
            survey_output_dir / "supplementary_evidence_report.md",
            supplement_agent.format_report_as_markdown(report),
        )
        return report, supplement_agent

    local_queries, web_queries = collect_supplement_queries(supplement_plan)

    local_candidates = retrieve_local_supplement_candidates(
        retrieval_agent=retrieval_agent,
        local_queries=local_queries,
        top_m=top_m,
        local_k=SUPPLEMENT_LOCAL_K,
        core_papers=paper_records,
    )
    save_json(local_candidates, survey_output_dir / "supplement_local_candidates.json")

    web_output = {
        "queries": web_queries,
        "engines": SUPPLEMENT_WEB_ENGINES,
        "limit_per_query": 0,
        "from_year": SUPPLEMENT_FROM_YEAR,
        "results": [],
        "errors": [],
    }

    if SUPPLEMENT_WEB_K > 0 and web_queries:
        limit_per_query = max(
            1,
            math.ceil(SUPPLEMENT_WEB_K / max(1, len(web_queries))),
        )
        web_searcher = WebLiteratureSearcher(
            ads_api_token=SUPPLEMENT_ADS_API_TOKEN,
            timeout=SUPPLEMENT_WEB_TIMEOUT,
        )
        web_output = web_searcher.search_many(
            queries=web_queries,
            engines=SUPPLEMENT_WEB_ENGINES,
            limit_per_query=limit_per_query,
            from_year=SUPPLEMENT_FROM_YEAR,
        )
        web_output["results"] = web_output.get("results", [])[:SUPPLEMENT_WEB_K]

    save_json(web_output, survey_output_dir / "supplement_web_results.json")

    if SUPPLEMENT_ENABLE_CANDIDATE_SCREENING:
        supplement_screening = supplement_agent.screen_supplement_candidates(
            query=query,
            survey_mode=survey_mode,
            supplement_plan=supplement_plan,
            local_candidates=local_candidates,
            web_candidates=web_output.get("results", []),
            accept_relevance_levels=SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS,
        )
        save_json(
            supplement_screening,
            survey_output_dir / "supplement_candidate_screening.json",
        )

        synthesis_local_candidates = supplement_screening.get(
            "accepted_local_candidates",
            [],
        )
        synthesis_web_candidates = supplement_screening.get(
            "accepted_web_candidates",
            [],
        )
        save_json(
            synthesis_local_candidates,
            survey_output_dir / "supplement_local_candidates_screened.json",
        )
        save_json(
            synthesis_web_candidates,
            survey_output_dir / "supplement_web_results_screened.json",
        )
    else:
        supplement_screening = {
            "enabled": False,
            "screening_results": [],
            "accepted_local_candidates": local_candidates,
            "accepted_web_candidates": web_output.get("results", []),
            "rejected_count": 0,
        }
        synthesis_local_candidates = local_candidates
        synthesis_web_candidates = web_output.get("results", [])
        save_json(
            supplement_screening,
            survey_output_dir / "supplement_candidate_screening.json",
        )

    supplement_report = supplement_agent.synthesize_supplement(
        query=query,
        paper_records=paper_records,
        survey_mode=survey_mode,
        supplement_plan=supplement_plan,
        local_candidates=synthesis_local_candidates,
        web_candidates=synthesis_web_candidates,
    )
    supplement_report["web_errors"] = web_output.get("errors", [])
    supplement_report["candidate_screening"] = {
        "enabled": supplement_screening.get("enabled"),
        "accepted_local_count": len(synthesis_local_candidates),
        "accepted_web_count": len(synthesis_web_candidates),
        "rejected_count": supplement_screening.get("rejected_count", 0),
    }
    save_json(supplement_report, survey_output_dir / "supplement_report.json")
    write_text(
        survey_output_dir / "supplementary_evidence_report.md",
        supplement_agent.format_report_as_markdown(supplement_report),
    )

    return supplement_report, supplement_agent


def main():
    args = parse_args()

    default_query = "Did an ancient ocean exist in the Chryse Planitia of Mars?"

    if args.query is None:
        query = input_query(default_query=default_query)
    else:
        query = args.query.strip()

    if args.top_m is None:
        top_m = input_top_m(default_top_m=DEFAULT_TOP_K)
    else:
        top_m = args.top_m

    if args.survey_mode is None:
        survey_mode = input_survey_mode()
    else:
        survey_mode = args.survey_mode

    if survey_mode == "lightweight":
        content_source = "metadata"
    elif args.content_source is None:
        content_source = input_content_source()
    else:
        content_source = args.content_source

    print("=" * 100)
    print("Literature Survey Stage With Optional Supplement")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Top-M: {top_m}")
    print(f"Batch size: {args.batch_size}")
    print(f"Survey mode: {survey_mode}")
    print(f"Content source: {content_source}")
    if content_source == "md":
        print(f"Markdown corpus dir: {args.md_corpus_dir}")
    print(f"Max PDF chars: {args.max_pdf_chars}")
    print(f"Use cache: {not args.no_cache}")
    print(f"Supplement enabled: {ENABLE_SUPPLEMENT_ROUND}")
    print(f"Supplement mode: {SUPPLEMENT_MODE}")

    print("\n" + "=" * 100)
    print("Stage 1: Retrieval")
    print("=" * 100)

    retrieval_agent = RetrievalAgent.from_config()

    top_m_papers = retrieval_agent.retrieve(
        query=query,
        top_k=top_m,
        normalize=True,
    )

    run_keyword = extract_run_keyword_from_query_plan(
        retrieval_agent.last_query_plan
    )

    run_output_dir, search_output_dir, survey_output_dir = build_run_output_dirs(
        project_root=PROJECT_ROOT,
        output_root_dir=args.output_root_dir,
        run_keyword=run_keyword,
        survey_mode=f"{survey_mode}_supplement",
        content_source=content_source,
    )

    top_m_path = search_output_dir / "top_m_papers.json"
    selected_path = survey_output_dir / "selected_papers.json"
    survey_path = survey_output_dir / "survey.txt"
    mindmap_path = survey_output_dir / "MindMap.txt"
    run_config_path = run_output_dir / "run_config.json"

    save_run_config(
        output_path=run_config_path,
        query=query,
        run_keyword=run_keyword,
        top_m=top_m,
        batch_size=args.batch_size,
        survey_mode=survey_mode,
        content_source=content_source,
        md_corpus_dir=args.md_corpus_dir,
        max_pdf_chars=args.max_pdf_chars,
        use_cache=not args.no_cache,
        run_output_dir=run_output_dir,
        search_output_dir=search_output_dir,
        survey_output_dir=survey_output_dir,
    )

    save_json(top_m_papers, top_m_path)

    print(retrieval_agent.format_query_plan())
    print(f"Run keyword: {run_keyword}")
    print(f"Run output dir: {run_output_dir}")
    print(f"Search output dir: {search_output_dir}")
    print(f"Survey output dir: {survey_output_dir}")
    print(f"Run config: {run_config_path}")
    print(f"Saved Top-M papers to: {top_m_path}")
    print(f"Retrieved papers: {len(top_m_papers)}")

    print("\n" + "=" * 100)
    print("Stage 2: Relevance Screening")
    print("=" * 100)

    relevance_agent = RelevanceAgent(
        temperature=0.0,
    )

    selected_papers = relevance_agent.screen_top_m(
        query=query,
        papers=top_m_papers,
        batch_size=args.batch_size,
    )

    selected_papers = sorted(selected_papers, key=safe_year)
    save_json(selected_papers, selected_path)

    print(f"Saved selected papers to: {selected_path}")
    print(f"Selected papers: {len(selected_papers)}")

    if len(selected_papers) == 0:
        print("No relevant papers were selected. Stop survey generation.")
        return

    print("\n" + "=" * 100)
    print("Stage 3: Build Literature Records")
    print("=" * 100)

    record_output_dir = ensure_dir(
        survey_output_dir / "paper_records" / survey_mode / content_source
    )

    paper_digest_agent = PaperDigestAgent(
        temperature=0.2,
        max_pdf_chars=args.max_pdf_chars,
        digest_cache_dir=str(record_output_dir),
        use_cache=not args.no_cache,
        content_source=content_source if content_source != "metadata" else "pdf",
        md_corpus_dir=args.md_corpus_dir,
    )

    if survey_mode == "lightweight":
        paper_records = paper_digest_agent.build_lightweight_records(
            selected_papers=selected_papers,
        )
        paper_records_path = survey_output_dir / "paper_records_lightweight.json"

    elif survey_mode == "middle":
        paper_records = paper_digest_agent.build_middle_records(
            query=query,
            selected_papers=selected_papers,
        )
        paper_records_path = survey_output_dir / "paper_records_middle.json"

    elif survey_mode == "full":
        paper_records = paper_digest_agent.build_full_records(
            query=query,
            selected_papers=selected_papers,
        )
        paper_records_path = survey_output_dir / "paper_records_full.json"

    else:
        raise ValueError(f"Unsupported survey mode: {survey_mode}")

    paper_records = sorted(paper_records, key=safe_year)
    save_json(paper_records, paper_records_path)

    print(f"Saved literature records to: {paper_records_path}")
    print(f"Paper records: {len(paper_records)}")

    print("\n" + "=" * 100)
    print("Stage 3.5: Optional Supplement Evidence Round")
    print("=" * 100)

    supplement_report, supplement_agent = run_supplement_round(
        query=query,
        survey_mode=survey_mode,
        top_m=top_m,
        paper_records=paper_records,
        retrieval_agent=retrieval_agent,
        survey_output_dir=survey_output_dir,
    )

    if supplement_report:
        print(f"Supplement effect: {supplement_report.get('overall_effect')}")
        print(f"Supplement summary: {supplement_report.get('summary')}")
    else:
        print("Supplement report is not attached to final survey.")

    print("\n" + "=" * 100)
    print("Stage 4: Final Survey Generation")
    print("=" * 100)

    survey_agent = SupplementedSurveyAgent(
        temperature=0.2,
    )

    survey_text = survey_agent.generate_and_save(
        query=query,
        paper_records=paper_records,
        survey_mode=survey_mode,
        output_path=str(survey_path),
        supplement_report=supplement_report,
        supplement_agent=supplement_agent,
    )

    print(f"Saved survey report to: {survey_path}")

    print("\n" + "=" * 100)
    print("Stage 5: Final MindMap Generation")
    print("=" * 100)

    mindmap_agent = MindMapAgent(
        temperature=0.2,
    )

    mindmap_agent.generate_and_save(
        query=query,
        survey_text=survey_text,
        paper_records=paper_records,
        survey_mode=survey_mode,
        output_path=str(mindmap_path),
        supplement_context=supplement_agent.format_report_for_mindmap(
            supplement_report
        )
        if supplement_agent is not None
        else "",
    )

    print(f"Saved mind map to: {mindmap_path}")

    print("\n" + "=" * 100)
    print("Survey stage with optional supplement finished.")
    print("=" * 100)
    print(f"Run output dir:      {run_output_dir}")
    print(f"Run config:          {run_config_path}")
    print(f"Top-M papers:        {top_m_path}")
    print(f"Selected papers:     {selected_path}")
    print(f"Literature records:  {paper_records_path}")
    print(f"Survey report:       {survey_path}")
    print(f"Mind map:            {mindmap_path}")
    print(f"Supplement evidence: {survey_output_dir / 'supplementary_evidence_report.md'}")


if __name__ == "__main__":
    main()
