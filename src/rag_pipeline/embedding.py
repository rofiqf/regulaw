"""Tahap 3 Embedding: vektorisasi teks secara lokal dengan sentence-transformers."""
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL


def get_embeddings(model_name=EMBEDDING_MODEL):
    print("[Agent] Inisialisasi model embedding lokal (mpnet-base-v2)...")
    return HuggingFaceEmbeddings(model_name=model_name)
