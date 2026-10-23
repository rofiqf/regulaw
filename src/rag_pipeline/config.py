"""Konfigurasi pipeline RAG dari environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").lower()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.200.6:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))


def configure_llm(provider=None, model=None, api_key=None):
    """Override konfigurasi LLM saat runtime (dipakai oleh web GUI)."""
    global LLM_PROVIDER, LM_STUDIO_MODEL, GROQ_MODEL, GEMINI_MODEL
    global LM_STUDIO_API_KEY, GROQ_API_KEY, GEMINI_API_KEY
    if provider:
        LLM_PROVIDER = provider.lower()
    if model:
        if LLM_PROVIDER == "gemini":
            GEMINI_MODEL = model
        elif LLM_PROVIDER == "groq":
            GROQ_MODEL = model
        else:
            LM_STUDIO_MODEL = model
    if api_key:
        if LLM_PROVIDER == "gemini":
            GEMINI_API_KEY = api_key
        elif LLM_PROVIDER == "groq":
            GROQ_API_KEY = api_key
        else:
            LM_STUDIO_API_KEY = api_key
