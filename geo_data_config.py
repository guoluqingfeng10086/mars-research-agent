"""Configuration for optional Mars geologic context retrieval.

This file is intentionally separate from config.py. The literature survey
pipeline does not depend on these paths.
"""

import os
from pathlib import Path


GEO_DATA_ENABLED = True

# Network retrieval for HiRISE can be slow or unavailable. Keep it configurable.
USE_HIRISE = True
USE_MINERAL_ABUNDANCE = True


# Override any path with environment variables if local directories differ.
GLOBAL_DATA_ROOT = Path(os.getenv("MARS_GLOBAL_DATA_ROOT", r"G:\Lenovo\全球数据"))
GEODATA_ROOT = Path(os.getenv("MARS_GEODATA_ROOT", r"G:\MMAgent\geodata"))

GEOLOGIC_SHAPEFILE_PATH = Path(
    os.getenv(
        "MARS_GEOLOGIC_SHAPEFILE_PATH",
        str(
            GLOBAL_DATA_ROOT
            / "(包含地质年代)SIM3292_MarsGlobalGeologicGIS_20M"
            / "SIM3292_Shapefiles"
            / "SIM3292_Global_Geology.shp"
        ),
    )
)

ALBEDO_TIF_PATH = Path(
    os.getenv(
        "MARS_ALBEDO_TIF_PATH",
        str(GLOBAL_DATA_ROOT / "OMEGA反照率" / "ALBEDO_R1080_EQU_MAP.tiff"),
    )
)

ELEVATION_TIF_PATH = Path(
    os.getenv(
        "MARS_ELEVATION_TIF_PATH",
        str(GLOBAL_DATA_ROOT / "地形高程" / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"),
    )
)

PALEOLAKE_CSV_PATH = Path(
    os.getenv("MARS_PALEOLAKE_CSV_PATH", str(GEODATA_ROOT / "2016（古湖泊）.csv"))
)

CRATER_CSV_PATH = Path(
    os.getenv("MARS_CRATER_CSV_PATH", str(GEODATA_ROOT / "craters1km.csv"))
)

VALLEY_SHP_PATH = Path(
    os.getenv(
        "MARS_VALLEY_SHP_PATH",
        str(GEODATA_ROOT / "火星河谷系统全球地图" / "Giulia Valleys.shp"),
    )
)

MINERAL_DB_PATH = Path(
    os.getenv("MARS_MINERAL_DB_PATH", str(GEODATA_ROOT / "mineral_abundance.db"))
)
