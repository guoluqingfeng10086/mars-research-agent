"""
Standalone web literature search demo.

This file does not touch the existing local retrieval pipeline. It can query
OpenAlex Works API, NASA ADS API, and Crossref REST API, normalize the returned
metadata, and print the results as JSON.

Recommended usage in PowerShell:

    python scripts/web_literature_search_demo.py `
      --query "Chryse Planitia Mars" `
      --from_year 2020 `
      --limit 5

If you run the script without arguments, it will ask for query, year, limit,
and engines in the terminal.

API keys are defined below for quick local testing. Command-line arguments and
environment variables can still override them.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_QUERY = "Chryse Planitia Mars"
DEFAULT_LIMIT = 15
DEFAULT_ENGINES = ["ads"]

# Quick local testing keys. Do not commit this file to a public repository with
# real credentials in it.
DEFAULT_OPENALEX_API_KEY = "9aFcLNNE0OoAzwT5OS5sUg"
DEFAULT_ADS_API_TOKEN = "ujGBAylmIlW8mM8pG5bEnY53Ze9DP8qoBrpGpSDq"


class LiteratureResult:
    def __init__(
        self,
        source: str,
        rank: int,
        title: str = "",
        year: str = "",
        journal: str = "",
        authors: str = "",
        doi: str = "",
        url: str = "",
        abstract: str = "",
        citation_count: Optional[int] = None,
        open_access_pdf_url: str = "",
        external_id: str = "",
        raw_score: Optional[float] = None,
    ):
        self.source = source
        self.rank = rank
        self.title = title
        self.year = year
        self.journal = journal
        self.authors = authors
        self.doi = doi
        self.url = url
        self.abstract = abstract
        self.citation_count = citation_count
        self.open_access_pdf_url = open_access_pdf_url
        self.external_id = external_id
        self.raw_score = raw_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "rank": self.rank,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "authors": self.authors,
            "doi": self.doi,
            "url": self.url,
            "abstract": self.abstract,
            "citation_count": self.citation_count,
            "open_access_pdf_url": self.open_access_pdf_url,
            "external_id": self.external_id,
            "raw_score": self.raw_score,
        }


def _request_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    retries: int = 2,
) -> Dict[str, Any]:
    headers = headers or {}
    headers.setdefault(
        "User-Agent",
        "MarsResearchAgent/0.1 (literature search demo; mailto:example@example.com)",
    )

    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset))
        except HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"Request failed: {last_error}")


def _safe_join(values: Any, sep: str = "; ") -> str:
    if not values:
        return ""

    if isinstance(values, str):
        return values

    if isinstance(values, list):
        return sep.join(str(v).strip() for v in values if str(v).strip())

    return str(values)


def _first(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0] or "")
    if value is None:
        return ""
    return str(value)


def _normalize_doi(value: Any) -> str:
    doi = _first(value).strip()
    doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.lower()


def _openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""

    positions = []
    for word, indexes in inverted_index.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            try:
                positions.append((int(index), str(word)))
            except Exception:
                continue

    if not positions:
        return ""

    return " ".join(word for _, word in sorted(positions))


def search_openalex(
    query: str,
    limit: int = 5,
    api_key: str = "",
    from_year: Optional[int] = None,
    timeout: int = 30,
) -> List[LiteratureResult]:
    params = {
        "search": query,
        "per_page": max(1, min(limit, 100)),
        "sort": "relevance_score:desc",
        "select": ",".join(
            [
                "id",
                "doi",
                "display_name",
                "publication_year",
                "authorships",
                "primary_location",
                "best_oa_location",
                "open_access",
                "abstract_inverted_index",
                "cited_by_count",
                "relevance_score",
            ]
        ),
    }

    if api_key:
        params["api_key"] = api_key

    if from_year is not None:
        params["filter"] = f"from_publication_date:{from_year}-01-01,type:article"

    url = "https://api.openalex.org/works?" + urlencode(params)
    data = _request_json(url, timeout=timeout)

    results = []
    for rank, item in enumerate(data.get("results", []), start=1):
        authors = []
        for authorship in item.get("authorships", []) or []:
            author = authorship.get("author", {}) if isinstance(authorship, dict) else {}
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        best_oa_location = item.get("best_oa_location") or {}

        results.append(
            LiteratureResult(
                source="openalex",
                rank=rank,
                title=item.get("display_name", "") or "",
                year=str(item.get("publication_year", "") or ""),
                journal=source.get("display_name", "") or "",
                authors=_safe_join(authors),
                doi=_normalize_doi(item.get("doi", "")),
                url=item.get("doi") or item.get("id", "") or "",
                abstract=_openalex_abstract(item.get("abstract_inverted_index")),
                citation_count=item.get("cited_by_count"),
                open_access_pdf_url=best_oa_location.get("pdf_url", "") or "",
                external_id=item.get("id", "") or "",
                raw_score=item.get("relevance_score"),
            )
        )

    return results[:limit]


def search_ads(
    query: str,
    limit: int = 5,
    token: str = "",
    from_year: Optional[int] = None,
    ads_query: str = "",
    timeout: int = 30,
) -> List[LiteratureResult]:
    if not token:
        raise ValueError(
            "NASA ADS search requires a token. Set ADS_API_TOKEN or pass --ads_api_token."
        )

    ads_query = ads_query.strip() if ads_query else query
    if from_year is not None:
        current_year = datetime.now().year
        ads_query = f"year:{from_year}-{current_year} AND ({ads_query})"

    params = {
        "q": ads_query,
        "rows": max(1, min(limit, 200)),
        "sort": "score desc",
        "fl": ",".join(
            [
                "bibcode",
                "title",
                "author",
                "year",
                "pub",
                "doi",
                "abstract",
                "citation_count",
                "identifier",
                "score",
            ]
        ),
    }

    url = "https://api.adsabs.harvard.edu/v1/search/query?" + urlencode(params)
    data = _request_json(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )

    docs = data.get("response", {}).get("docs", [])
    results = []

    for rank, item in enumerate(docs, start=1):
        bibcode = item.get("bibcode", "") or ""
        doi = _normalize_doi(item.get("doi", ""))

        results.append(
            LiteratureResult(
                source="ads",
                rank=rank,
                title=_first(item.get("title", "")),
                year=str(item.get("year", "") or ""),
                journal=item.get("pub", "") or "",
                authors=_safe_join(item.get("author", [])),
                doi=doi,
                url=f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract" if bibcode else "",
                abstract=item.get("abstract", "") or "",
                citation_count=item.get("citation_count"),
                open_access_pdf_url="",
                external_id=bibcode,
                raw_score=item.get("score"),
            )
        )

    return results[:limit]


def _crossref_year(item: Dict[str, Any]) -> str:
    for field in ["published-print", "published-online", "published"]:
        date_parts = item.get(field, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return str(date_parts[0][0])
    return ""


def _crossref_authors(item: Dict[str, Any]) -> str:
    authors = []
    for author in item.get("author", []) or []:
        given = author.get("given", "")
        family = author.get("family", "")
        name = " ".join(part for part in [given, family] if part).strip()
        if name:
            authors.append(name)
    return _safe_join(authors)


def search_crossref(
    query: str,
    limit: int = 5,
    from_year: Optional[int] = None,
    timeout: int = 30,
) -> List[LiteratureResult]:
    params = {
        "query.bibliographic": query,
        "rows": max(1, min(limit, 100)),
        "sort": "relevance",
        "order": "desc",
        "select": ",".join(
            [
                "DOI",
                "title",
                "author",
                "published-print",
                "published-online",
                "published",
                "container-title",
                "URL",
                "abstract",
                "is-referenced-by-count",
                "type",
            ]
        ),
    }

    if from_year is not None:
        params["filter"] = f"from-pub-date:{from_year}-01-01"
    url = "https://api.crossref.org/works?" + urlencode(params)
    data = _request_json(url, timeout=timeout)
    items = data.get("message", {}).get("items", [])
    results = []
    for rank, item in enumerate(items, start=1):
        results.append(
            LiteratureResult(
                source="crossref",
                rank=rank,
                title=_first(item.get("title", "")),
                year=_crossref_year(item),
                journal=_first(item.get("container-title", "")),
                authors=_crossref_authors(item),
                doi=_normalize_doi(item.get("DOI", "")),
                url=item.get("URL", "") or "",
                abstract=item.get("abstract", "") or "",
                citation_count=item.get("is-referenced-by-count"),
                open_access_pdf_url="",
                external_id=_normalize_doi(item.get("DOI", "")),
                raw_score=None,
            )
        )

    return results[:limit]


def search_all(
    query: str,
    engines: List[str],
    limit: int,
    openalex_api_key: str = "",
    ads_api_token: str = "",
    from_year: Optional[int] = None,
    ads_query: str = "",
    timeout: int = 30,
) -> Dict[str, Any]:
    all_results: Dict[str, Any] = {
        "query": query,
        "ads_query": ads_query,
        "limit_per_engine": limit,
        "from_year": from_year,
        "engines": engines,
        "results": [],
        "errors": [],
    }

    for engine in engines:
        try:
            if engine == "openalex":
                results = search_openalex(
                    query=query,
                    limit=limit,
                    api_key=openalex_api_key,
                    from_year=from_year,
                    timeout=timeout,
                )
            elif engine == "ads":
                results = search_ads(
                    query=query,
                    limit=limit,
                    token=ads_api_token,
                    from_year=from_year,
                    ads_query=ads_query,
                    timeout=timeout,
                )
            elif engine == "crossref":
                results = search_crossref(
                    query=query,
                    limit=limit,
                    from_year=from_year,
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Unsupported engine: {engine}")

            all_results["results"].extend(item.to_dict() for item in results)

        except Exception as exc:
            all_results["errors"].append(
                {
                    "engine": engine,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search web literature APIs and print normalized metadata."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Literature search query. Default is ADS-friendly keywords.",
    )
    parser.add_argument(
        "--ads_query",
        default="",
        help=(
            "Optional raw NASA ADS query. Use this for ADS syntax such as "
            "'abs:\"Chryse Planitia\" AND (abs:ocean OR abs:delta)'."
        ),
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["openalex", "ads", "crossref"],
        default=DEFAULT_ENGINES,
        help="Search engines to use. Defaults to NASA ADS.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of records to fetch from each selected engine.",
    )
    parser.add_argument(
        "--from_year",
        type=int,
        default=None,
        help="Only search literature published from this year onward, e.g. 2020.",
    )
    parser.add_argument(
        "--openalex_api_key",
        default=os.getenv("OPENALEX_API_KEY", DEFAULT_OPENALEX_API_KEY),
        help="OpenAlex API key. Defaults to script constant, overridden by OPENALEX_API_KEY env var.",
    )
    parser.add_argument(
        "--ads_api_token",
        default=os.getenv("ADS_API_TOKEN", DEFAULT_ADS_API_TOKEN),
        help="NASA ADS API token. Defaults to script constant, overridden by ADS_API_TOKEN env var.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def _prompt_text(label: str, default: str = "") -> str:
    if default:
        value = input(f"{label} [{default}]: ").strip()
        return value if value else default

    return input(f"{label}: ").strip()


def _prompt_optional_int(label: str, default: Optional[int] = None) -> Optional[int]:
    while True:
        if default is None:
            raw = input(f"{label} [all years]: ").strip()
        else:
            raw = input(f"{label} [{default}]: ").strip()

        if not raw:
            return default

        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer, or press Enter to keep the default.")


def _prompt_positive_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()

        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            print("Please enter a positive integer.")
            continue

        if value <= 0:
            print("Please enter a positive integer.")
            continue

        return value


def _prompt_engines(default: List[str]) -> List[str]:
    default_text = " ".join(default)
    prompt = (
        "Engines: ads / openalex / crossref / all "
        f"[{default_text}]: "
    )

    while True:
        raw = input(prompt).strip().lower()

        if not raw:
            return list(default)

        values = raw.replace(",", " ").split()

        if values == ["all"]:
            return ["ads", "openalex", "crossref"]

        allowed = {"ads", "openalex", "crossref"}
        invalid = [value for value in values if value not in allowed]

        if invalid:
            print(f"Unsupported engine(s): {', '.join(invalid)}")
            continue

        return values


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print("=" * 80)
    print("Web literature search demo")
    print("Press Enter to keep the value shown in brackets.")
    print("=" * 80)

    args.query = _prompt_text("Query", args.query)
    args.from_year = _prompt_optional_int("From year", args.from_year)
    args.limit = _prompt_positive_int("Limit per engine", args.limit)
    args.engines = _prompt_engines(args.engines)

    print("=" * 80)
    print("Running search...")
    print("=" * 80)

    return args


def main() -> int:
    args = parse_args()

    if len(sys.argv) == 1:
        args = configure_interactively(args)

    output = search_all(
        query=args.query,
        engines=args.engines,
        limit=args.limit,
        openalex_api_key=args.openalex_api_key,
        ads_api_token=args.ads_api_token,
        from_year=args.from_year,
        ads_query=args.ads_query,
        timeout=args.timeout,
    )

    output_text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"

    try:
        sys.stdout.flush()
        sys.stdout.buffer.write(output_text.encode("utf-8"))
    except AttributeError:
        print(output_text)

    if output["errors"]:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
