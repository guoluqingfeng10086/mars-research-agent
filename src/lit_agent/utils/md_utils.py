# src/lit_agent/utils/md_utils.py

from pathlib import Path
from typing import Optional, Union


def _windows_long_path(path: Path) -> str:
    path_str = str(path.resolve())

    if not path_str.startswith("\\\\?\\") and len(path_str) >= 260:
        return "\\\\?\\" + path_str

    return path_str


def resolve_md_path(file_name: str, md_root: Union[str, Path]) -> Optional[Path]:
    if not file_name:
        return None

    md_root = Path(md_root)
    stem = Path(file_name).stem.strip()

    if not md_root.exists():
        return None

    exact_root_path = md_root / f"{stem}.md"
    if exact_root_path.exists():
        return exact_root_path

    return None


def extract_md_text(md_path: Union[str, Path]) -> str:
    md_path = Path(md_path)

    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
    except FileNotFoundError:
        try:
            with open(_windows_long_path(md_path), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Markdown not found: {md_path}")

    if not text:
        raise ValueError(f"No text extracted from Markdown: {md_path}")

    return text
