"""Tahap 5 Retrieval: hibrida vektor semantik + BM25 (keyword) untuk presisi yuridis."""
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from .config import TOP_K

VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5


def create_retriever(vector_store, k=TOP_K):
    return vector_store.as_retriever(search_kwargs={"k": k})


def create_hybrid_retriever(chunks, vector_store, k=TOP_K):
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT],
    )


def retrieve_context(retriever, query):
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)
