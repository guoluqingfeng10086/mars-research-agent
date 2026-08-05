# src/lit_agent/utils/io_utils.py

import json
from pathlib import Path
from typing import Any, Iterable, List, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Union[str, Path], default: str = "") -> str:
    path = Path(path)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable objects into JSON-safe objects.
    This handles numpy arrays, numpy scalar types, Path objects, tuples, etc.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(k): _to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            _to_jsonable(v)
            for v in obj
        ]

    # numpy support
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    return str(obj)


def save_json(data: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _to_jsonable(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    path = Path(path)

    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunk_list(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]
