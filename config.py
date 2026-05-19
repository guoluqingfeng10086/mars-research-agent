# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CORPUS_PKL_PATH = str(PROJECT_ROOT / "mmcorpus_bge.pkl")
MODEL_PATH = r"G:\model\bge-large-en-v1.5"
PDF_CORPUS_DIR = r"G:\MMCorpus2000v1"
MD_CORPUS_DIR = r"G:\MMCorpus2000v1_md"
DEFAULT_MAX_SOURCE_CHARS = 150000

# outputs
OUTPUT_ROOT_DIR = "outputs"

DEVICE = "cuda"
MAX_SEQ_LENGTH = 512

DEFAULT_TOP_K = 30

# Vector field weights
RESEARCH_CONTENT_WEIGHT = 0.7
ABSTRACT_NOTE_WEIGHT = 0.3

# Parallel recall size
VECTOR_RECALL_K = 300
BM25_RECALL_K = 300

# RRF fusion
RRF_K = 80

# Phrase boost after RRF
PHRASE_BOOST_WEIGHT = 0.35

# =========================
# LLM query analyzer config
# =========================

USE_LLM_QUERY_ANALYZER = True


USE_LLM_QUERY_ANALYZER = True

QUERY_LLM_API_KEY = "sk-94YIEkzyU4YfiNdVQa1puFXrbWqN3NtRmNdeE7nNlo1M5E0r"
QUERY_LLM_BASE_URL = "https://api.nuwaflux.com/v1"
QUERY_LLM_MODEL = "gpt-4o-mini"
QUERY_LLM_TIMEOUT = 60
