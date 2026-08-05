# scripts/run_discovery_agent.py

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_TOP_K


def parse_evidence_sources(value: str):
    sources = [item.strip().lower() for item in str(value or "").split(",")]
    sources = [item for item in sources if item]
    allowed = {"survey", "local", "web", "geo"}
    invalid = [item for item in sources if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            "Unsupported evidence sources: " + ", ".join(invalid)
        )
    if not sources:
        raise argparse.ArgumentTypeError("At least one evidence source is required.")
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the general discovery workflow over survey memory, literature, "
            "and optional geologic products."
        )
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Scientific question to answer. If omitted, input interactively.",
    )
    parser.add_argument("--top-n-dirs", type=int, default=10)
    parser.add_argument("--min-reuse-score", type=float, default=0.35)
    parser.add_argument("--top-m", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--survey-mode",
        choices=["lightweight", "middle", "full"],
        default="full",
    )
    parser.add_argument(
        "--content-source",
        choices=["pdf", "md"],
        default="md",
    )
    parser.add_argument(
        "--no-new-survey",
        action="store_true",
        help="Fail instead of running a new survey when no reusable output is found.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "auto",
            "generic",
            "detection_confidence",
            "hypothesis_evaluation",
            "mineral_confidence",
        ],
        default="auto",
        help=(
            "Final assessment mode. mineral_confidence is retained as a "
            "backward-compatible alias for detection_confidence."
        ),
    )
    parser.add_argument(
        "--evidence-sources",
        type=parse_evidence_sources,
        default=parse_evidence_sources("survey,local,web,geo"),
        help="Comma-separated sources: survey,local,web,geo.",
    )
    parser.add_argument(
        "--no-gap-filling",
        action="store_true",
        help="Evaluate completeness but do not acquire additional evidence.",
    )
    parser.add_argument(
        "--max-discovery-rounds",
        type=int,
        default=2,
        help="Maximum evidence-acquisition rounds after the initial completeness check.",
    )
    parser.add_argument(
        "--max-gaps-per-round",
        type=int,
        default=3,
        help="Maximum high-value evidence gaps handled in one round.",
    )
    parser.add_argument("--local-k", type=int, default=5)
    parser.add_argument("--web-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from lit_agent.agents.discovery_agent import DiscoveryAgent

    question = args.question.strip() if args.question else ""
    if not question:
        question = input("Please enter the scientific question:\n> ").strip()
    if not question:
        raise ValueError("Scientific question cannot be empty.")

    agent = DiscoveryAgent.from_config()
    result = agent.run(
        question=question,
        top_n_dirs=args.top_n_dirs,
        min_reuse_score=args.min_reuse_score,
        top_m=args.top_m,
        survey_mode=args.survey_mode,
        content_source=args.content_source,
        run_survey_if_missing=not args.no_new_survey,
        prompt_mode=args.prompt_mode,
        evidence_sources=args.evidence_sources,
        enable_gap_filling=not args.no_gap_filling,
        max_discovery_rounds=args.max_discovery_rounds,
        max_gaps_per_round=args.max_gaps_per_round,
        local_k=args.local_k,
        web_k=args.web_k,
    )

    print("=" * 100)
    print("Discovery agent finished.")
    print("=" * 100)
    print(f"Matched run: {result['matched_run_name']}")
    print(f"Match score: {result['match_score']}")
    print(f"Created new survey: {result['created_new_survey']}")
    print(f"Used geo context: {result['used_geo_context']}")
    print(f"Prompt mode: {result['prompt_mode']}")
    print(f"Evidence sources: {', '.join(result['evidence_sources'])}")
    print(f"Discovery rounds: {result['discovery_round_count']}")
    print(f"Stop reason: {result['stop_reason']}")
    print(f"Answerability: {result['completeness']['answerability']}")
    print(f"Output dir: {result['output_dir']}")
    print(f"Report: {Path(result['output_dir']) / 'discovery_report.txt'}")


if __name__ == "__main__":
    main()
