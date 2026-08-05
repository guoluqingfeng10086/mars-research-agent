# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_PKL_PATH = str(PROJECT_ROOT / "mmcorpus_bge_merged.pkl")
# bge-large-en-v1.5 path
MODEL_PATH = ""


# =========================
# LLM config
# =========================
USE_LLM_QUERY_ANALYZER = True
QUERY_LLM_API_KEY = ""
QUERY_LLM_BASE_URL = ""
QUERY_LLM_MODEL = ""
QUERY_LLM_TIMEOUT = 60

# paper path
PDF_CORPUS_DIR = ""
MD_CORPUS_DIR = ""
DEFAULT_MAX_SOURCE_CHARS = 150000

# outputs
OUTPUT_ROOT_DIR = "outputs"

DEVICE = "cuda"
MAX_SEQ_LENGTH = 512
DEFAULT_TOP_K = 30

# =========================
# Optional supplement round
# =========================
# These settings are only used by scripts/run_survey_with_supplement.py.
# The original search_demo.py and run_survey_stage.py flows do not read them.
ENABLE_SUPPLEMENT_ROUND = True
SUPPLEMENT_MODE = "auto"  # off | auto | force
SUPPLEMENT_LOCAL_K = 10
SUPPLEMENT_WEB_K = 10
SUPPLEMENT_MAX_GAPS = 2
SUPPLEMENT_MAX_WEB_QUERIES_PER_GAP = 2
SUPPLEMENT_WEB_ENGINES = ["ads", "crossref"]
SUPPLEMENT_FROM_YEAR = None
SUPPLEMENT_WEB_TIMEOUT = 30
SUPPLEMENT_ADS_API_TOKEN = ""
SUPPLEMENT_ENABLE_CANDIDATE_SCREENING = True
SUPPLEMENT_ACCEPT_RELEVANCE_LEVELS = ["high", "medium"]

# Vector field weights
RESEARCH_CONTENT_WEIGHT = 0.7
ABSTRACT_NOTE_WEIGHT = 0.3

# Parallel recall size
VECTOR_RECALL_K = 300
BM25_RECALL_K = 300

# RRF fusion
RRF_K = 80

# Phrase boost after RRF
PHRASE_BOOST_WEIGHT = 0.45


