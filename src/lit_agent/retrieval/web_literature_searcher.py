import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class WebLiteratureResult:
    def __init__(
        self,
        source: str,
        query: str,
        rank: int,
        title: str = "",
        year: str = "",
        journal: str = "",
        authors: str = "",
        doi: str = "",
        url: str = "",
        abstract: str = "",
        citation_count: Optional[int] = None,
        external_id: str = "",
        raw_score: Optional[float] = None,
    ):
        self.source = source
        self.query = query
        self.rank = rank
        self.title = title
        self.year = year
        self.journal = journal
        self.authors = authors
        self.doi = doi
        self.url = url
        self.abstract = abstract
        self.citation_count = citation_count
        self.external_id = external_id
        self.raw_score = raw_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "query": self.query,
            "rank": self.rank,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "authors": self.authors,
            "doi": self.doi,
            "url": self.url,
            "abstract": self.abstract,
            "citation_count": self.citation_count,
            "external_id": self.external_id,
            "raw_score": self.raw_score,
        }


class WebLiteratureSearcher:
    """
    Thin ADS/Crossref metadata searcher for the supplement flow.

    It returns title/abstract/metadata only. It does not fetch full text and does
    not rank against local corpus results.
    """

    def __init__(
        self,
        ads_api_token: str = "",
        timeout: int = 30,
        retries: int = 2,
        user_agent: str = "ResearchAgent/0.1 (metadata supplement search)",
    ):
        self.ads_api_token = ads_api_token or os.getenv("ADS_API_TOKEN", "")
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent

    def _request_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        request_headers = dict(headers or {})
        request_headers.setdefault("User-Agent", self.user_agent)

        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                request = Request(url, headers=request_headers)
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return json.loads(response.read().decode(charset))
            except HTTPError as exc:
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except URLError as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

        raise RuntimeError(f"Request failed: {last_error}")

    def _first(self, value: Any) -> str:
        if isinstance(value, list) and value:
            return str(value[0] or "")
        if value is None:
            return ""
        return str(value)

    def _safe_join(self, values: Any, sep: str = "; ") -> str:
        if not values:
            return ""
        if isinstance(values, str):
            return values
        if isinstance(values, list):
            return sep.join(str(v).strip() for v in values if str(v).strip())
        return str(values)

    def _normalize_doi(self, value: Any) -> str:
        doi = self._first(value).strip()
        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://dx.doi.org/", "")
        return doi.lower()

    def _crossref_year(self, item: Dict[str, Any]) -> str:
        for field in ["published-print", "published-online", "published"]:
            date_parts = item.get(field, {}).get("date-parts", [])
            if date_parts and date_parts[0]:
                return str(date_parts[0][0])
        return ""

    def _crossref_authors(self, item: Dict[str, Any]) -> str:
        authors = []
        for author in item.get("author", []) or []:
            given = author.get("given", "")
            family = author.get("family", "")
            name = " ".join(part for part in [given, family] if part).strip()
            if name:
                authors.append(name)
        return self._safe_join(authors)

    def search_ads(
        self,
        query: str,
        limit: int = 5,
        from_year: Optional[int] = None,
    ) -> List[WebLiteratureResult]:
        if not self.ads_api_token:
            raise ValueError(
                "NASA ADS search requires ADS_API_TOKEN or SUPPLEMENT_ADS_API_TOKEN."
            )

        ads_query = query.strip()
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
        data = self._request_json(
            url,
            headers={"Authorization": f"Bearer {self.ads_api_token}"},
        )

        docs = data.get("response", {}).get("docs", [])
        results = []

        for rank, item in enumerate(docs, start=1):
            bibcode = item.get("bibcode", "") or ""
            doi = self._normalize_doi(item.get("doi", ""))

            results.append(
                WebLiteratureResult(
                    source="ads",
                    query=query,
                    rank=rank,
                    title=self._first(item.get("title", "")),
                    year=str(item.get("year", "") or ""),
                    journal=item.get("pub", "") or "",
                    authors=self._safe_join(item.get("author", [])),
                    doi=doi,
                    url=(
                        f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
                        if bibcode
                        else ""
                    ),
                    abstract=item.get("abstract", "") or "",
                    citation_count=item.get("citation_count"),
                    external_id=bibcode,
                    raw_score=item.get("score"),
                )
            )

        return results[:limit]

    def search_crossref(
        self,
        query: str,
        limit: int = 5,
        from_year: Optional[int] = None,
    ) -> List[WebLiteratureResult]:
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
        data = self._request_json(url)
        items = data.get("message", {}).get("items", [])
        results = []

        for rank, item in enumerate(items, start=1):
            results.append(
                WebLiteratureResult(
                    source="crossref",
                    query=query,
                    rank=rank,
                    title=self._first(item.get("title", "")),
                    year=self._crossref_year(item),
                    journal=self._first(item.get("container-title", "")),
                    authors=self._crossref_authors(item),
                    doi=self._normalize_doi(item.get("DOI", "")),
                    url=item.get("URL", "") or "",
                    abstract=item.get("abstract", "") or "",
                    citation_count=item.get("is-referenced-by-count"),
                    external_id=self._normalize_doi(item.get("DOI", "")),
                    raw_score=None,
                )
            )

        return results[:limit]

    def search_one(
        self,
        query: str,
        engines: List[str],
        limit: int,
        from_year: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for engine in engines:
            try:
                if engine == "ads":
                    engine_results = self.search_ads(
                        query=query,
                        limit=limit,
                        from_year=from_year,
                    )
                elif engine == "crossref":
                    engine_results = self.search_crossref(
                        query=query,
                        limit=limit,
                        from_year=from_year,
                    )
                else:
                    raise ValueError(f"Unsupported web literature engine: {engine}")

                results.extend(item.to_dict() for item in engine_results)

            except Exception as exc:
                errors.append(
                    {
                        "query": query,
                        "engine": engine,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        return results, errors

    def search_many(
        self,
        queries: List[str],
        engines: List[str],
        limit_per_query: int,
        from_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        all_results: List[Dict[str, Any]] = []
        all_errors: List[Dict[str, str]] = []

        for query in queries:
            query = str(query or "").strip()
            if not query:
                continue

            results, errors = self.search_one(
                query=query,
                engines=engines,
                limit=limit_per_query,
                from_year=from_year,
            )
            all_results.extend(results)
            all_errors.extend(errors)

        return {
            "queries": queries,
            "engines": engines,
            "limit_per_query": limit_per_query,
            "from_year": from_year,
            "results": self._dedupe_results(all_results),
            "errors": all_errors,
        }

    def _dedupe_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []

        for item in results:
            doi = str(item.get("doi", "") or "").lower().strip()
            title = str(item.get("title", "") or "").lower().strip()
            year = str(item.get("year", "") or "").strip()
            key = doi or f"{title}|{year}"

            if not key or key in seen:
                continue

            seen.add(key)
            deduped.append(item)

        return deduped
