"""Coordinate parsing helpers for Mars science questions."""

import re
from typing import Optional, Set


class Coordinate:
    def __init__(self, lat: float, lon: float, raw_text: str = ""):
        self.lat = lat
        self.lon = lon
        self.raw_text = raw_text

    @property
    def lon_180(self) -> float:
        lon = ((self.lon + 180.0) % 360.0) - 180.0
        return 180.0 if lon == -180.0 and self.lon > 0 else lon

    @property
    def lon_360(self) -> float:
        return self.lon_180 % 360.0

    def as_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "lon_180": self.lon_180,
            "lon_360": self.lon_360,
            "raw_text": self.raw_text,
        }


_NUM = r"[-+]?\d+(?:\.\d+)?"


def _apply_hemisphere(value: float, hemi: str, is_lat: bool) -> float:
    hemi = (hemi or "").upper()
    if hemi in {"S", "W"}:
        return -abs(value)
    if hemi in {"N", "E"}:
        return abs(value)
    return value


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -360.0 <= lon <= 360.0


def extract_coordinate(text: str) -> Optional[Coordinate]:
    """Extract the first lat/lon pair from a question.

    Supported examples:
    - 25.4 N, 63.2 E
    - 25.4°N, 63.2°E
    - lat 25.4 lon 63.2
    - latitude: 25.4, longitude: -30.35
    - 25.4, 63.2
    """
    text = text or ""

    labelled = re.search(
        rf"(?:lat|latitude)\s*[:=]?\s*(?P<lat>{_NUM})\s*[° ]*(?P<lat_h>[NS])?.*?"
        rf"(?:lon|lng|longitude)\s*[:=]?\s*(?P<lon>{_NUM})\s*[° ]*(?P<lon_h>[EW])?",
        text,
        flags=re.IGNORECASE,
    )
    if labelled:
        lat = _apply_hemisphere(
            float(labelled.group("lat")), labelled.group("lat_h") or "", True
        )
        lon = _apply_hemisphere(
            float(labelled.group("lon")), labelled.group("lon_h") or "", False
        )
        if _valid(lat, lon):
            return Coordinate(lat=lat, lon=lon, raw_text=labelled.group(0))

    hemispheric = re.search(
        rf"(?P<lat>{_NUM})\s*[° ]*(?P<lat_h>[NS])\s*[,; ]+\s*"
        rf"(?P<lon>{_NUM})\s*[° ]*(?P<lon_h>[EW])",
        text,
        flags=re.IGNORECASE,
    )
    if hemispheric:
        lat = _apply_hemisphere(
            float(hemispheric.group("lat")), hemispheric.group("lat_h"), True
        )
        lon = _apply_hemisphere(
            float(hemispheric.group("lon")), hemispheric.group("lon_h"), False
        )
        if _valid(lat, lon):
            return Coordinate(lat=lat, lon=lon, raw_text=hemispheric.group(0))

    chinese = re.search(
        rf"(?:北纬|纬度)\s*(?P<lat>{_NUM}).*?(?:东经|经度)\s*(?P<lon>{_NUM})",
        text,
    )
    if chinese:
        lat = float(chinese.group("lat"))
        lon = float(chinese.group("lon"))
        if _valid(lat, lon):
            return Coordinate(lat=lat, lon=lon, raw_text=chinese.group(0))

    pair = re.search(rf"(?P<lat>{_NUM})\s*,\s*(?P<lon>{_NUM})", text)
    if pair:
        lat = float(pair.group("lat"))
        lon = float(pair.group("lon"))
        if _valid(lat, lon):
            return Coordinate(lat=lat, lon=lon, raw_text=pair.group(0))

    return None


def tokenize_for_match(text: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    keep_short = {"mg", "fe", "ca", "al", "h2o", "co2"}
    stop = {
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "what",
        "how",
        "are",
        "is",
        "do",
        "does",
        "did",
        "mars",
        "martian",
    }
    return {t for t in tokens if (len(t) >= 3 or t in keep_short) and t not in stop}
