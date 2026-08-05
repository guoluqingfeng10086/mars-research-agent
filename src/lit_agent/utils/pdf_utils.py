# src/lit_agent/utils/pdf_utils.py

from pathlib import Path
from typing import Optional, Union


def resolve_pdf_path(file_name: str, pdf_root: Union[str, Path]) -> Optional[Path]:
    if not file_name:
        return None

    pdf_root = Path(pdf_root)

    direct_path = pdf_root / file_name
    if direct_path.exists():
        return direct_path

    matches = list(pdf_root.rglob(file_name))
    if matches:
        return matches[0]

    return None


def extract_pdf_text(pdf_path: Union[str, Path]) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required. Install it with: pip install pymupdf"
        ) from exc

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    texts = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        if text:
            texts.append(text)

    doc.close()

    return "\n\n".join(texts).strip()
