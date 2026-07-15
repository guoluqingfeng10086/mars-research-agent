# scripts/run_batch_discovery_validation.py

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_TOP_K
from lit_agent.agents.discovery_agent import DiscoveryAgent, SurveyRunCandidate
from lit_agent.retrieval.geo_context_retriever import query_and_summarize_geo_context


REQUIRED_COLUMNS = ["CenterLat", "CenterLon", "name"]
DEFAULT_INPUT_TABLE = Path(
    "C:\\Users\\Lenovo\\Desktop\\ai4research\\"
    "\u63a2\u6d4b\u7ed3\u679c_\u4e4c\u6258\u90a6\u5e73\u539f"
    "\u4e3b\u8981\u77ff\u7269.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-validate detection confidence from a spreadsheet. The script "
            "adds one confidence column for local geologic context only and one "
            "for local geologic context plus literature/survey evidence."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_TABLE),
        help="Input .xlsx/.xls/.csv table.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output table path. Defaults to a timestamped file next to the input.",
    )
    parser.add_argument(
        "--report-root",
        default=str(PROJECT_ROOT / "discovery_stage" / "batch_validation"),
        help="Directory for per-row reports and JSON results.",
    )
    parser.add_argument("--sheet-name", default=0, help="Excel sheet name or index.")
    parser.add_argument("--limit", type=int, default=None, help="Only process N rows.")
    parser.add_argument("--start-row", type=int, default=0, help="Zero-based row offset.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between rows.")
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
        help="For literature mode, fail instead of running a new survey when no reusable output is found.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows that already have both confidence columns filled.",
    )
    return parser.parse_args()


def normalize_sheet_name(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def read_table(path: Path, sheet_name) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and openpyxl are required for batch table I/O.") from exc

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_table(df, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        df.to_excel(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    raise ValueError(f"Unsupported output format: {path.suffix}")


def default_output_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_path.with_name(f"{input_path.stem}_discovery_validation_1_{timestamp}.xlsx")


def validate_columns(df) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def validate_runtime_dependencies() -> None:
    missing = []
    for package_name, import_name in [
        ("openai", "openai"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        python_exe = sys.executable
        install_cmd = f'"{python_exe}" -m pip install ' + " ".join(missing)
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + "\nInstall them in the same environment used to run this script:\n"
            + install_cmd
        )


def material_label(name: Any) -> str:
    text = str(name or "").strip()
    replacements = {
        "h2o": "H2O",
        "fe": "Fe",
        "mg": "Mg",
    }
    parts = [replacements.get(part.lower(), part) for part in text.split()]
    return " ".join(parts) or "target material"


def make_question(material: str, lat: float, lon: float, mode: str) -> str:
    base = (
        f"Classify the confidence of {material} detection in Utopia Planitia, "
        f"Mars, at {lat:.4f} deg N, {lon:.4f} deg E into levels 1-4 "
        f"(levels 1-2: reliable; levels 3-4: uncertain or low-confidence)"
    )
    if mode == "geo_only":
        return (
            base
            + ", based on geomorphic evidence and local geologic context."
        )
    return (
        base
        + ", based on geomorphic evidence, local geologic context, and findings from published literature."
    )


def extract_confidence(report: str) -> str:
    if not report:
        return ""
    patterns = [
        r"\bLevel\s*([1-4])\b",
        r"\blevel\s*([1-4])\b",
        r"\b([1-4])\s*级\b",
        r"置信度\s*[:：]?\s*([1-4])",
    ]
    for pattern in patterns:
        match = re.search(pattern, report)
        if match:
            return f"Level {match.group(1)}"
    return "Unparsed"


def extract_consistency(report: str, label: str) -> str:
    if not report:
        return ""
    label_pattern = re.escape(label).replace("\\-", "[- ]")
    pattern = (
        rf"{label_pattern}\s*[:：]?\s*"
        r"(consistent|partly consistent|weak|inconsistent|not provided)"
    )
    match = re.search(pattern, report, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "Unparsed"


def extract_label_value(report: str, label: str, allowed_values) -> str:
    if not report:
        return ""
    label_pattern = re.escape(label).replace("\\-", "[- ]")
    values_pattern = "|".join(re.escape(value) for value in allowed_values)
    pattern = rf"{label_pattern}\s*[:：]?\s*({values_pattern})"
    match = re.search(pattern, report, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "Unparsed"


def extract_local_fit(report: str) -> str:
    return extract_label_value(
        report,
        "Local geologic fit",
        ["partial fit", "weak fit", "fit", "mismatch"],
    )


def extract_literature_fit(report: str) -> str:
    return extract_label_value(
        report,
        "Literature contextual fit",
        [
            "direct local",
            "same geologic setting",
            "same region",
            "broad analogy",
            "irrelevant",
            "not provided",
        ],
    )


def extract_conflict_level(report: str) -> str:
    return extract_label_value(
        report,
        "Conflict or alternative explanation",
        ["dominant alternative", "significant", "minor", "none"],
    )


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def safe_name(text: str, max_len: int = 70) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', " ", str(text or ""))
    text = re.sub(r"\s+", "_", text).strip("_")
    text = re.sub(r"[^A-Za-z0-9_\-.]+", "", text)
    return (text or "row")[:max_len]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def run_geo_only_report(
    agent: DiscoveryAgent,
    question: str,
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    geo_context, geo_summary = query_and_summarize_geo_context(lat=lat, lon=lon)
    matched = SurveyRunCandidate(
        path=PROJECT_ROOT,
        name="local_geologic_context_only",
        query="Local geologic context only.",
        score=0.0,
    )
    report = agent._generate_report(
        question=question,
        matched_run=matched,
        survey_text="Not provided for this local-geologic-context-only run.",
        mindmap_text="Not provided for this local-geologic-context-only run.",
        geo_summary=geo_summary or "No local geologic product context was available.",
        prompt_mode="mineral_confidence",
    )
    return {
        "question": question,
        "matched_run_name": matched.name,
        "matched_query": matched.query,
        "match_score": matched.score,
        "created_new_survey": False,
        "coordinates": {"lat": lat, "lon": lon},
        "used_geo_context": bool(geo_summary),
        "geo_context": geo_context,
        "discovery_report": report,
    }


def run_discovery_report(
    agent: DiscoveryAgent,
    question: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return agent.run(
        question=question,
        top_n_dirs=args.top_n_dirs,
        min_reuse_score=args.min_reuse_score,
        top_m=args.top_m,
        survey_mode=args.survey_mode,
        content_source=args.content_source,
        run_survey_if_missing=not args.no_new_survey,
        prompt_mode="mineral_confidence",
    )


def row_report_dir(root: Path, row_index: int, material: str, lat: float, lon: float) -> Path:
    label = safe_name(f"row_{row_index:05d}_{material}_{lat:.4f}_{lon:.4f}")
    return root / label


def process_row(
    df,
    index: int,
    agent: DiscoveryAgent,
    args: argparse.Namespace,
    report_root: Path,
) -> Tuple[str, str]:
    lat = safe_float(df.at[index, "CenterLat"])
    lon = safe_float(df.at[index, "CenterLon"])
    material = material_label(df.at[index, "name"])
    if lat is None or lon is None:
        raise ValueError("CenterLat or CenterLon is empty or invalid.")

    out_dir = row_report_dir(report_root, index, material, lat, lon)

    geo_question = make_question(material, lat, lon, mode="geo_only")
    geo_result = run_geo_only_report(agent, geo_question, lat, lon)
    geo_report = geo_result.get("discovery_report", "")
    geo_confidence = extract_confidence(geo_report)
    geo_local_fit = extract_local_fit(geo_report)
    geo_conflict = extract_conflict_level(geo_report)

    lit_question = make_question(material, lat, lon, mode="geo_literature")
    lit_result = run_discovery_report(agent, lit_question, args)
    lit_report = lit_result.get("discovery_report", "")
    lit_confidence = extract_confidence(lit_report)
    lit_local_fit = extract_local_fit(lit_report)
    lit_literature_fit = extract_literature_fit(lit_report)
    lit_conflict = extract_conflict_level(lit_report)

    write_text(out_dir / "geo_only_question.txt", geo_question)
    write_text(out_dir / "geo_only_report.txt", geo_report)
    write_text(out_dir / "geo_literature_question.txt", lit_question)
    write_text(out_dir / "geo_literature_report.txt", lit_report)
    save_json(
        out_dir / "result.json",
        {
            "row_index": index,
            "center_lat": lat,
            "center_lon": lon,
            "name": material,
            "geo_only_confidence": geo_confidence,
            "geo_literature_confidence": lit_confidence,
            "geo_only_local_fit": geo_local_fit,
            "geo_only_conflict": geo_conflict,
            "geo_literature_local_fit": lit_local_fit,
            "geo_literature_contextual_fit": lit_literature_fit,
            "geo_literature_conflict": lit_conflict,
            "geo_result": geo_result,
            "literature_result": lit_result,
        },
    )

    df.at[index, "geo_only_confidence"] = geo_confidence
    df.at[index, "geo_literature_confidence"] = lit_confidence
    df.at[index, "geo_only_local_fit"] = geo_local_fit
    df.at[index, "geo_only_conflict"] = geo_conflict
    df.at[index, "geo_literature_local_fit"] = lit_local_fit
    df.at[index, "geo_literature_contextual_fit"] = lit_literature_fit
    df.at[index, "geo_literature_conflict"] = lit_conflict
    return geo_confidence, lit_confidence


def main() -> None:
    args = parse_args()
    validate_runtime_dependencies()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    report_root = Path(args.report_root) / datetime.now().strftime("%Y%m%d_%H%M%S")

    df = read_table(input_path, normalize_sheet_name(args.sheet_name))
    validate_columns(df)

    for column in [
        "geo_only_confidence",
        "geo_literature_confidence",
        "geo_only_local_fit",
        "geo_only_conflict",
        "geo_literature_local_fit",
        "geo_literature_contextual_fit",
        "geo_literature_conflict",
    ]:
        if column not in df.columns:
            df[column] = ""

    agent = DiscoveryAgent.from_config()
    row_indices = list(df.index)[args.start_row :]
    if args.limit is not None:
        row_indices = row_indices[: args.limit]

    for count, index in enumerate(row_indices, start=1):
        if args.skip_existing:
            if str(df.at[index, "geo_only_confidence"]).strip() and str(
                df.at[index, "geo_literature_confidence"]
            ).strip():
                continue
        try:
            geo_level, lit_level = process_row(df, index, agent, args, report_root)
            print(f"[{count}/{len(row_indices)}] row {index}: geo={geo_level}, geo+lit={lit_level}")
        except Exception as exc:
            print(f"[{count}/{len(row_indices)}] row {index}: ERROR {exc}")

        write_table(df, output_path)
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_table(df, output_path)
    print(f"Batch validation finished: {output_path}")
    print(f"Reports: {report_root}")


if __name__ == "__main__":
    main()
