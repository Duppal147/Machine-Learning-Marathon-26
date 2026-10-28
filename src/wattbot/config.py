"""Shared settings, read from environment variables (see README for the .env setup).

Variable names match the original MVP's .env.example, so existing .env files keep working.
"""
import os

# BadgerBrain gateway (UW-Madison; requires GlobalProtect VPN)
BASE_URL = os.environ.get("BADGERBRAIN_BASE_URL", "https://llm-gw01.doit.wisc.edu/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
REQUEST_TIMEOUT = 300  # first call after a cold start can take ~90s

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-vl-embedding-8b")
EMBEDDING_DIM = 4096

# Vision-language chat model that reads chart images (scripts/describe_figures.py). Empty means
# "pick the gateway's only Qwen-VL chat model"; run scripts/check_vision_model.py to see what is served.
VISION_MODEL = os.environ.get("VISION_MODEL", "")

# Vector store (under the gitignored documents/ folder)
CHROMA_DIR = os.environ.get("CHROMA_DIR", "documents/chroma")

# Qwen3 embeddings are asymmetric: queries get this instruction, documents are embedded as-is.
QUERY_INSTRUCTION = "Given a question, retrieve passages that answer the question"
