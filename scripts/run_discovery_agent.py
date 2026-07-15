# scripts/run_discovery_agent.py

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_TOP_K
from lit_agent.agents.discovery_agent import DiscoveryAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the discovery agent over existing survey outputs."
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
        choices=["auto", "generic", "mineral_confidence"],
        default="auto",
        help="Prompt mode for report generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    )

    print("=" * 100)
    print("Discovery agent finished.")
    print("=" * 100)
    print(f"Matched run: {result['matched_run_name']}")
    print(f"Match score: {result['match_score']}")
    print(f"Created new survey: {result['created_new_survey']}")
    print(f"Used geo context: {result['used_geo_context']}")
    print(f"Prompt mode: {result['prompt_mode']}")
    print(f"Output dir: {result['output_dir']}")
    print(f"Report: {Path(result['output_dir']) / 'discovery_report.txt'}")


if __name__ == "__main__":
    main()
