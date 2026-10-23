"""Tahap 4 Vector Store: pengindeksan vektor efisien dengan FAISS."""
from langchain_community.vectorstores import FAISS


def build_index(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)
