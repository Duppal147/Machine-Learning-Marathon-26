import os

BASE_URL = os.environ.get("BADGERBRAIN_BASE_URL", "https://llm-gw01.doit.wisc.edu/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.8-27b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-vl-embedding-8b")
VISION_MODEL = os.environ.get("VISION_MODEL", "churro-3b")

EMBEDDING_DIM = 4096
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
PDF_CACHE_DIR = os.environ.get("PDF_CACHE_DIR", "./pdf_cache")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

REQUEST_TIMEOUT = 300

TOP_K_RETRIEVAL = 15
TOP_K_RERANK = 5
NUM_QUERY_EXPANSIONS = 3

SUBMISSION_COLUMNS = [
    "id", "question", "answer", "answer_value", "answer_unit",
    "ref_id", "ref_url", "supporting_materials", "explanation",
]
