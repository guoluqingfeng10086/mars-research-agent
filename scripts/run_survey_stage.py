# scripts/run_survey_stage.py
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer project source code.
# These two lines must appear before importing config or lit_agent modules.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

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

from lit_agent.agents.retrieval_agent import RetrievalAgent
from lit_agent.agents.relevance_agent import RelevanceAgent
from lit_agent.agents.paper_digest_agent import PaperDigestAgent
from lit_agent.agents.survey_agent import SurveyAgent
from lit_agent.agents.mindmap_agent import MindMapAgent
from lit_agent.agents.research_map_agent import ResearchMapAgent
from lit_agent.utils.io_utils import ensure_dir, save_json


def sanitize_keyword_for_dir(keyword: str, max_len: int = 60) -> str:
    """
    Convert keyword text into a Windows-safe short folder name.
    """
    text = str(keyword).strip()

    text = re.sub(r'[\\/:*?"<>|]+', " ", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "", text)
    text = text.strip("_")

    return text[:max_len]


def _as_list(value):
    """
    Convert query-plan fields into a clean list.
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    return [str(value).strip()] if str(value).strip() else []


def extract_run_keyword_from_query_plan(query_plan) -> Optional[str]:
    """
    Use the already computed QueryAnalyzer result to build the run keyword.

    Priority:
    1. target_regions
    2. geological_concepts
    3. query_phrases

    If no usable keyword is found, return None.
    """
    if query_plan is None:
        return None

    target_regions = _as_list(getattr(query_plan, "target_regions", None))
    geological_concepts = _as_list(getattr(query_plan, "geological_concepts", None))
    query_phrases = _as_list(getattr(query_plan, "query_phrases", None))

    selected_terms = []

    # Usually most suitable for folder name.
    selected_terms.extend(target_regions[:2])

    # Add one or two concept terms if available.
    selected_terms.extend(geological_concepts[:2])

    # Fallback to query phrases.
    if not selected_terms:
        selected_terms.extend(query_phrases[:3])

    # Remove duplicates while preserving order.
    seen = set()
    unique_terms = []

    for term in selected_terms:
        norm = term.lower()
        if norm in seen:
            continue
        seen.add(norm)
        unique_terms.append(term)

    if not unique_terms:
        return None

    keyword = "_".join(unique_terms)
    keyword = sanitize_keyword_for_dir(keyword)

    return keyword if keyword else None


def build_run_output_dirs(
    project_root: Path,
    output_root_dir: str,
    run_keyword: Optional[str] = None,
    survey_mode: Optional[str] = None,
    content_source: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Build this structure:

    outputs/
        <timestamp>_<keyword>/
            search_stage/
            survey_stage/

    If keyword is empty:
        outputs/
            <timestamp>/
                search_stage/
                survey_stage/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_keyword:
        keyword = sanitize_keyword_for_dir(run_keyword)
    else:
        keyword = ""

    name_parts = []
    if keyword:
        name_parts.append(keyword)
    name_parts.append(timestamp)
    if content_source and content_source != "metadata":
        name_parts.append(content_source)
    if survey_mode:
        name_parts.append(survey_mode)

    run_dir_name = "_".join(name_parts)

    output_root = Path(output_root_dir)

    if not output_root.is_absolute():
        output_root = project_root / output_root

    run_output_dir = output_root / run_dir_name
    search_output_dir = run_output_dir / "search_stage"
    survey_output_dir = run_output_dir / "survey_stage"

    search_output_dir.mkdir(parents=True, exist_ok=True)
    survey_output_dir.mkdir(parents=True, exist_ok=True)

    return run_output_dir, search_output_dir, survey_output_dir


def save_run_config(
    output_path: Path,
    query: str,
    run_keyword: Optional[str],
    top_m: int,
    batch_size: int,
    survey_mode: str,
    content_source: str,
    md_corpus_dir: str,
    max_pdf_chars,
    use_cache: bool,
    run_output_dir: Path,
    search_output_dir: Path,
    survey_output_dir: Path,
):
    """
    Save run metadata for reproducibility.
    """
    config = {
        "query": query,
        "run_keyword": run_keyword,
        "top_m": top_m,
        "batch_size": batch_size,
        "survey_mode": survey_mode,
        "content_source": content_source,
        "md_corpus_dir": md_corpus_dir,
        "max_pdf_chars": max_pdf_chars,
        "use_cache": use_cache,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_output_dir": str(run_output_dir),
        "search_output_dir": str(search_output_dir),
        "survey_output_dir": str(survey_output_dir),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run literature survey stage."
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
        help=(
            "Root output directory. A run subfolder will be created under it. "
            "Default is OUTPUT_ROOT_DIR in config.py or 'outputs'."
        ),
    )

    parser.add_argument(
        "--max_pdf_chars",
        type=int,
        default=None,
        help=(
            "Maximum characters read from each PDF in middle/full mode. "
            "Default None means use full extracted PDF text."
        ),
    )

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable paper record cache.",
    )

    return parser.parse_args()


def input_query(default_query: Optional[str] = None) -> str:
    print("\nPlease enter your scientific question.")
    if default_query:
        print(f"Default: {default_query}")

    query = input("> ").strip()

    if query:
        return query

    if default_query:
        return default_query

    raise ValueError("Query cannot be empty.")


def input_top_m(default_top_m: int = 30) -> int:
    print("\nPlease enter Top-M retrieved papers.")
    print(f"Default: {default_top_m}")

    value = input("> ").strip()

    if not value:
        return default_top_m

    try:
        top_m = int(value)
    except ValueError:
        raise ValueError(f"Invalid Top-M value: {value}")

    if top_m <= 0:
        raise ValueError("Top-M must be positive.")

    return top_m


def input_survey_mode() -> str:
    print("\nPlease select survey mode:")
    print("1. lightweight - use abstract_note / research_content only; fast")
    print("2. middle      - read PDFs for high + medium papers; generate query_centered_content")
    print("3. full        - medium -> query_centered_content; high -> expert-level deep analysis")

    value = input("> ").strip().lower()

    if value in ["1", "lightweight", "light"]:
        return "lightweight"

    if value in ["2", "middle", "mid"]:
        return "middle"

    if value in ["3", "full"]:
        return "full"

    raise ValueError(f"Invalid survey mode: {value}")


def input_content_source() -> str:
    print("\nPlease select content source:")
    print("1. pdf - read original PDF files")
    print("2. md  - read Markdown files")

    value = input("> ").strip().lower()

    if value in ["1", "pdf"]:
        return "pdf"

    if value in ["2", "md", "markdown"]:
        return "md"

    raise ValueError(f"Invalid content source: {value}")


def safe_year(paper: dict) -> int:
    try:
        return int(float(paper.get("year", 9999)))
    except Exception:
        return 9999


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
    print("Literature Survey Stage")
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
        survey_mode=survey_mode,
        content_source=content_source,
    )

    top_m_path = search_output_dir / "top_m_papers.json"
    selected_path = survey_output_dir / "selected_papers.json"
    survey_path = survey_output_dir / "survey.txt"
    mindmap_path = survey_output_dir / "MindMap.txt"
    research_map_path = survey_output_dir / "research_map.json"
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
    print("Stage 4: Final Survey Generation")
    print("=" * 100)

    survey_agent = SurveyAgent(
        temperature=0.2,
    )

    survey_text = survey_agent.generate_and_save(
        query=query,
        paper_records=paper_records,
        survey_mode=survey_mode,
        output_path=str(survey_path),
    )

    print(f"Saved survey report to: {survey_path}")

    print("\n" + "=" * 100)
    print("Stage 5: Final MindMap Generation")
    print("=" * 100)

    mindmap_agent = MindMapAgent(
        temperature=0.2,
    )

    mindmap_text = mindmap_agent.generate_and_save(
        query=query,
        survey_text=survey_text,
        paper_records=paper_records,
        survey_mode=survey_mode,
        output_path=str(mindmap_path),
    )

    print(f"Saved mind map to: {mindmap_path}")

    print("\n" + "=" * 100)
    print("Stage 6: Structured Research Map Generation")
    print("=" * 100)

    research_map_agent = ResearchMapAgent(temperature=0.0)
    research_map_agent.generate_and_save(
        question=query,
        survey_text=survey_text,
        mindmap_text=mindmap_text,
        paper_records=paper_records,
        output_path=research_map_path,
    )
    print(f"Saved research map to: {research_map_path}")

    print("\n" + "=" * 100)
    print("Survey stage finished.")
    print("=" * 100)
    print(f"Run output dir:      {run_output_dir}")
    print(f"Run config:          {run_config_path}")
    print(f"Top-M papers:        {top_m_path}")
    print(f"Selected papers:     {selected_path}")
    print(f"Literature records:  {paper_records_path}")
    print(f"Survey report:       {survey_path}")
    print(f"Mind map:            {mindmap_path}")
    print(f"Research map:        {research_map_path}")


if __name__ == "__main__":
    main()
