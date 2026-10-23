"""Tahap 5 Retrieval: pengambilan top-k potongan dengan kemiripan semantik tertinggi."""
from .config import TOP_K


def create_retriever(vector_store, k=TOP_K):
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_context(retriever, query):
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)
