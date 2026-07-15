"""Optional Mars geologic product retrieval and summarization."""

import math
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geo_data_config import (
    ALBEDO_TIF_PATH,
    CRATER_CSV_PATH,
    ELEVATION_TIF_PATH,
    GEO_DATA_ENABLED,
    GEOLOGIC_SHAPEFILE_PATH,
    HTTP_PROXY,
    HTTPS_PROXY,
    MINERAL_DB_PATH,
    PALEOLAKE_CSV_PATH,
    USE_HIRISE,
    USE_MINERAL_ABUNDANCE,
    VALLEY_SHP_PATH,
)


R_MARS = 3396190.0
DEG2RAD = math.pi / 180.0


def _import_geo_deps() -> Dict[str, Any]:
    try:
        import fiona
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import rasterio
        import requests
        from geopy.distance import geodesic
        from lxml import etree
        from rasterio.transform import rowcol
        from shapely.geometry import Point, shape
        import shapely.geometry as geometry
    except Exception as exc:
        raise RuntimeError(
            "Geologic context dependencies are unavailable. Install pandas, "
            "geopandas, rasterio, shapely, fiona, geopy, requests, and lxml."
        ) from exc

    return locals()


def _path_exists(path: Path) -> bool:
    return bool(path) and Path(path).exists()


def _setup_proxy() -> None:
    if HTTP_PROXY:
        os.environ.setdefault("HTTP_PROXY", HTTP_PROXY)
        os.environ.setdefault("http_proxy", HTTP_PROXY)
    if HTTPS_PROXY:
        os.environ.setdefault("HTTPS_PROXY", HTTPS_PROXY)
        os.environ.setdefault("https_proxy", HTTPS_PROXY)


@lru_cache(maxsize=1)
def _load_geologic_dataset():
    deps = _import_geo_deps()
    gpd = deps["gpd"]
    data = gpd.read_file(GEOLOGIC_SHAPEFILE_PATH)
    data.sindex
    return data


def get_geologic_epoch(lon: float, lat: float) -> Optional[str]:
    if not _path_exists(GEOLOGIC_SHAPEFILE_PATH):
        return None
    deps = _import_geo_deps()
    geometry = deps["geometry"]
    data = _load_geologic_dataset()
    point = geometry.Point(lon, lat)
    possible = data.iloc[list(data.sindex.intersection(point.bounds))]
    for _, row in possible.iterrows():
        if row["geometry"].contains(point):
            return row.get("UnitDesc", None)
    return None


def _mars_lonlat_to_meters(lon_deg: float, lat_deg: float, radius: int = 3396000):
    return radius * math.radians(lon_deg), radius * math.radians(lat_deg)


def _find_nearest_valid(data, mask, row: int, col: int, max_radius: int = 10):
    for radius in range(1, max_radius + 1):
        row_min = max(0, row - radius)
        row_max = min(data.shape[0], row + radius + 1)
        col_min = max(0, col - radius)
        col_max = min(data.shape[1], col + radius + 1)
        window = data[row_min:row_max, col_min:col_max]
        window_mask = mask[row_min:row_max, col_min:col_max]
        valid = window[window_mask > 0]
        if valid.size > 0:
            return valid[0]
    return None


@lru_cache(maxsize=1)
def _load_raster(path: Path):
    deps = _import_geo_deps()
    return deps["rasterio"].open(path)


def get_albedo_value(lon: float, lat: float) -> Optional[float]:
    if not _path_exists(ALBEDO_TIF_PATH):
        return None
    deps = _import_geo_deps()
    rowcol = deps["rowcol"]
    scaling_factor = 1.4522365285e-05
    offset = 0.52414565669
    x, y = _mars_lonlat_to_meters(lon, lat)
    src = _load_raster(ALBEDO_TIF_PATH)
    row, col = rowcol(src.transform, x, y)
    if not (0 <= row < src.height and 0 <= col < src.width):
        return None
    data = src.read(1)
    mask = src.dataset_mask()
    raw_value = data[row, col]
    if mask[row, col] == 0:
        raw_value = _find_nearest_valid(data, mask, row, col)
        if raw_value is None:
            return None
    return float(raw_value * scaling_factor + offset)


def get_mars_elevation(lon: float, lat: float) -> Optional[float]:
    if not _path_exists(ELEVATION_TIF_PATH):
        return None
    deps = _import_geo_deps()
    rasterio = deps["rasterio"]
    rowcol = deps["rowcol"]
    src = _load_raster(ELEVATION_TIF_PATH)
    row, col = rowcol(src.transform, lon, lat)
    if not (0 <= row < src.height and 0 <= col < src.width):
        return None
    window = rasterio.windows.Window(col, row, 1, 1)
    mask_val = src.read_masks(1, window=window)[0, 0]
    if mask_val == 0:
        return None
    return float(src.read(1, window=window)[0, 0])


class HiRISESearcher:
    base_url = "https://www.uahirise.org/"
    search_url = "https://www.uahirise.org/results.php"

    def get_response(self, lon: float, lat: float, delta: float):
        deps = _import_geo_deps()
        requests = deps["requests"]
        params = {
            "lon_beg": str(max(0, lon - delta)),
            "lon_end": str(min(360, lon + delta)),
            "lat_beg": str(max(-90, lat - delta)),
            "lat_end": str(min(90, lat + delta)),
            "solar_all": "true",
            "image_all": "true",
            "order": "WP.release_date",
        }
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.uahirise.org/anazitisi.php"}
        return requests.get(self.search_url, params=params, headers=headers, timeout=20)

    def get_info(self, response) -> List[Dict[str, str]]:
        deps = _import_geo_deps()
        etree = deps["etree"]
        html_con = etree.HTML(response.content)
        results = []
        for td in html_con.xpath('//*[@class="catalog-cell-images"]'):
            item = {}
            href = td.xpath("a/@href")
            if href:
                item["id"] = href[0]
                item["url"] = self.base_url + href[0]
            desc = td.xpath("a/*/@alt")
            if desc:
                item["desc"] = desc[0]
            if item:
                results.append(item)
        return results


@lru_cache(maxsize=128)
def get_hirise_context(lat: float, lon: float):
    if not USE_HIRISE:
        return None, [], []
    _setup_proxy()
    lon_360 = lon % 360.0
    searcher = HiRISESearcher()
    for delta in [0.3, 0.4, 0.5]:
        try:
            results = searcher.get_info(searcher.get_response(lon_360, lat, delta))
            if results:
                return delta, results, results[:3]
        except Exception:
            break
    return None, [], []


@lru_cache(maxsize=1)
def _load_paleolake_csv():
    deps = _import_geo_deps()
    return deps["pd"].read_csv(PALEOLAKE_CSV_PATH)


def get_paleolake_context(lat: float, lon: float, delta: float = 2.0) -> List[Dict[str, Any]]:
    if not _path_exists(PALEOLAKE_CSV_PATH):
        return []
    pd = _import_geo_deps()["pd"]
    df = _load_paleolake_csv()
    lon_360 = lon % 360.0
    nearby = df[
        df["Lat. (N)"].between(lat - delta, lat + delta)
        & df["Lon. (E)"].between(lon_360 - delta, lon_360 + delta)
    ]
    basin_type = {"CBL": "closed-basin lake", "OBL": "open-basin lake"}
    valley_type = {"II": "isolated inlet valley", "VN": "valley network"}
    lakes = []
    for _, row in nearby.iterrows():
        lake = {
            "Basin Type": basin_type.get(row.get("Basin Type"), row.get("Basin Type")),
            "Lat": row.get("Lat. (N)"),
            "Lon": row.get("Lon. (E)"),
            "Valley Type": valley_type.get(row.get("Valley Type"), row.get("Valley Type")),
            "Degradation": row.get("Basin Degradation State"),
            "Strahler Order": row.get("Strahler Order"),
        }
        if pd.notna(row.get("Strahler Order Reference")):
            lake["Strahler Reference"] = row.get("Strahler Order Reference")
        lakes.append(lake)
    return lakes


@lru_cache(maxsize=1)
def _load_crater_csv():
    deps = _import_geo_deps()
    return deps["pd"].read_csv(CRATER_CSV_PATH, low_memory=False)


def get_crater_context(lat: float, lon: float, delta: float = 1.0) -> List[Dict[str, Any]]:
    if not _path_exists(CRATER_CSV_PATH):
        return []
    deps = _import_geo_deps()
    geodesic = deps["geodesic"]
    pd = deps["pd"]
    crater_df = _load_crater_csv()
    lon_360 = lon % 360.0
    nearby = crater_df[
        crater_df["LAT_CIRC_IMG"].between(lat - delta, lat + delta)
        & crater_df["LON_CIRC_IMG"].between(lon_360 - delta, lon_360 + delta)
    ].copy()
    if nearby.empty:
        return []
    center = (lat, lon_360)
    nearby = nearby.dropna(subset=["LAT_CIRC_IMG", "LON_CIRC_IMG"])
    nearby["distance"] = nearby.apply(
        lambda row: geodesic(center, (row["LAT_CIRC_IMG"], row["LON_CIRC_IMG"])).km,
        axis=1,
    )
    result = []
    for _, row in nearby.sort_values(by="distance").head(3).iterrows():
        result.append(
            {
                "crater_id": row.get("CRATER_ID"),
                "lat": row.get("LAT_CIRC_IMG"),
                "lon": row.get("LON_CIRC_IMG"),
                "diameter_km": row.get("DIAM_CIRC_IMG"),
                "int_morph1": row.get("INT_MORPH1"),
                "lay_morph1": row.get("LAY_MORPH1"),
                "DEG_RIM": row.get("DEG_RIM"),
                "DEG_EJC": row.get("DEG_EJC"),
                "DEG_FLR": row.get("DEG_FLR"),
            }
        )
    return result


@lru_cache(maxsize=1)
def _load_valley_shapefile():
    deps = _import_geo_deps()
    fiona = deps["fiona"]
    gpd = deps["gpd"]
    shape = deps["shape"]
    records = []
    with fiona.open(VALLEY_SHP_PATH, "r") as src:
        for feat in src:
            geom = feat.get("geometry")
            try:
                if not geom:
                    continue
                geom_obj = shape(geom)
                if geom_obj.geom_type not in ["LineString", "MultiLineString"]:
                    continue
                points = (
                    list(geom_obj.coords)
                    if geom_obj.geom_type == "LineString"
                    else [pt for part in geom_obj.geoms for pt in part.coords]
                )
                if len(points) > 1:
                    feat["geometry"] = geom_obj
                    records.append(feat)
            except Exception:
                continue
        return gpd.GeoDataFrame(
            [r["properties"] for r in records],
            geometry=[r["geometry"] for r in records],
            crs=src.crs,
        )


def get_valley_context(lat: float, lon: float, bins_km: Optional[List[int]] = None):
    if not _path_exists(VALLEY_SHP_PATH):
        return []
    bins_km = bins_km or [0, 20, 100]
    deps = _import_geo_deps()
    Point = deps["Point"]
    df = _load_valley_shapefile().copy()
    x0 = R_MARS * lon * DEG2RAD
    y0 = R_MARS * lat * DEG2RAD
    pt = Point(x0, y0)
    df["dist_km"] = df.geometry.distance(pt) / 1000.0
    df["lon_centroid"] = df.geometry.centroid.x / R_MARS / DEG2RAD
    df["lat_centroid"] = df.geometry.centroid.y / R_MARS / DEG2RAD
    if "Age" in df.columns:
        age_mapping = {"Hesp. Noac.": "Noachian-Hesperian", "Amaz. Hesp.": "Hesperian-Amazonian"}
        df["age_std"] = df["Age"].map(age_mapping).fillna(df["Age"])
    matched = []
    columns = ["dist_km", "lat_centroid", "lon_centroid", "Length(km)", "Age", "age_std", "Type"]
    for low, high in zip(bins_km[:-1], bins_km[1:]):
        subset = df[(df["dist_km"] > low) & (df["dist_km"] <= high)]
        if not subset.empty:
            matched.append((f"> {low} km and <= {high} km", subset[columns]))
    return matched


def get_mineral_abundance(lat: float, lon: float):
    if not USE_MINERAL_ABUNDANCE or not _path_exists(MINERAL_DB_PATH):
        return None, None
    lon_180 = ((lon + 180.0) % 360.0) - 180.0
    pixel_size = 0.25
    row = int((lat + 90.0) // pixel_size)
    col = int((lon_180 + 180.0) // pixel_size)
    idx = row * int(360 / pixel_size) + col + 1
    conn = sqlite3.connect(MINERAL_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT Amphibole, Dust, Hematite, High_Si_Glass, High_Ca_Px,
               K_Feldspar, Low_Ca_Px, Olivine, Plagioclase, Quartz
        FROM mineral_abundance
        WHERE idx = ?
        """,
        (idx,),
    )
    result = cur.fetchone()
    if not result:
        return idx, None
    minerals = [
        "Amphibole",
        "Dust",
        "Hematite",
        "High_Si_Glass",
        "High_Ca_Px",
        "K_Feldspar",
        "Low_Ca_Px",
        "Olivine",
        "Plagioclase",
        "Quartz",
    ]
    mineral_dict = {k: float(v) for k, v in zip(minerals, result) if v != -1}
    return idx, mineral_dict or None


def query_geo_context(lat: float, lon: float) -> Dict[str, Any]:
    if not GEO_DATA_ENABLED:
        return {"enabled": False}

    timings = {}

    def timed(name: str, fn, default):
        t0 = time.time()
        try:
            value = fn()
        except Exception as exc:
            value = default
            errors[name] = str(exc)
        timings[name] = round(time.time() - t0, 3)
        return value

    errors = {}  # type: Dict[str, str]
    lon_180 = ((lon + 180.0) % 360.0) - 180.0

    epoch = timed("geologic_epoch", lambda: get_geologic_epoch(lon_180, lat), None)
    albedo = timed("albedo", lambda: get_albedo_value(lon_180, lat), None)
    elevation = timed("elevation", lambda: get_mars_elevation(lon_180, lat), None)
    hirise_delta, hirise_all, hirise_top3 = timed(
        "hirise", lambda: get_hirise_context(lat, lon_180), (None, [], [])
    )
    paleolakes = timed("paleolake", lambda: get_paleolake_context(lat, lon_180), [])
    craters = timed("craters", lambda: get_crater_context(lat, lon_180), [])
    valley_groups = timed("valley", lambda: get_valley_context(lat, lon_180), [])
    mineral_idx, mineral_data = timed(
        "mineral", lambda: get_mineral_abundance(lat, lon_180), (None, None)
    )

    return {
        "enabled": True,
        "lat": lat,
        "lon": lon,
        "lon_180": lon_180,
        "lon_360": lon_180 % 360.0,
        "epoch": epoch,
        "albedo": albedo,
        "elevation": elevation,
        "hirise_all": hirise_all,
        "hirise_top3": hirise_top3,
        "hirise_delta": hirise_delta,
        "paleolakes": paleolakes,
        "craters": craters,
        "valley_groups": valley_groups,
        "mineral_idx": mineral_idx,
        "mineral_data": mineral_data,
        "timings": timings,
        "errors": errors,
    }


def summarize_geo_context(context: Dict[str, Any]) -> str:
    if not context or not context.get("enabled", True):
        return ""

    lines = [
        f"Location: lat {context.get('lat')}, lon {context.get('lon_180')} "
        f"({context.get('lon_360')}E)."
    ]

    if context.get("epoch"):
        lines.append(f"Regional geologic unit/epoch: {context['epoch']}.")
    if context.get("albedo") is not None:
        lines.append(f"Albedo value: approximately {context['albedo']:.3f}.")
    if context.get("elevation") is not None:
        lines.append(f"Elevation: {context['elevation']:.1f} m.")

    if context.get("hirise_all"):
        lines.append(
            f"Within +/-{context.get('hirise_delta')} degrees, "
            f"{len(context['hirise_all'])} HiRISE images were retrieved."
        )
        for i, img in enumerate(context["hirise_all"][:10], start=1):
            lines.append(f"  - HiRISE {i}: {img.get('desc', 'Unknown')} ({img.get('url', '')})")

    if context.get("paleolakes"):
        lines.append(f"Within +/-2 degrees, {len(context['paleolakes'])} paleolakes were found.")
        for lake in context["paleolakes"][:10]:
            lines.append(
                f"  - {lake.get('Basin Type')}, valley type {lake.get('Valley Type')}, "
                f"degradation {lake.get('Degradation')}."
            )

    if context.get("craters"):
        lines.append(f"{len(context['craters'])} nearby impact craters were found.")
        for crater in context["craters"]:
            lines.append(
                f"  - ID {crater.get('crater_id')}, diameter {crater.get('diameter_km')} km, "
                f"interior {crater.get('int_morph1')}, ejecta degradation {crater.get('DEG_EJC')}."
            )

    if context.get("valley_groups"):
        total = sum(len(group[1]) for group in context["valley_groups"])
        lines.append(f"{total} valleys were found in configured distance bins.")
        for label, df in context["valley_groups"]:
            lines.append(f"  - Range {label}: {len(df)} valleys.")
            for _, row in df.head(10).iterrows():
                lines.append(
                    f"    Type {row.get('Type')}, length {row.get('Length(km)')} km, "
                    f"age {row.get('age_std')}, distance {row.get('dist_km'):.2f} km."
                )

    if context.get("mineral_data"):
        lines.append("Estimated mineral abundances at this location:")
        for key, value in context["mineral_data"].items():
            lines.append(f"  - {key}: {value:.2f}")

    if context.get("errors"):
        lines.append("Unavailable geologic products:")
        for key, value in context["errors"].items():
            lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def query_and_summarize_geo_context(lat: float, lon: float) -> Tuple[Dict[str, Any], str]:
    context = query_geo_context(lat=lat, lon=lon)
    return context, summarize_geo_context(context)
