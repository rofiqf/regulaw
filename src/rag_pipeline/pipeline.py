"""Orkestrasi pipeline RAG 6 tahap untuk ekstraksi Literature GAP Matrix."""
import glob
import os

from . import config
from .chunking import split_documents
from .document_processing import load_pdf
from .embedding import get_embeddings
from .generation import GAP_MATRIX_QUERY, build_llm, generate
from .retrieval import create_hybrid_retriever, retrieve_context
from .self_healing import heal
from .vector_store import build_index


class GapMatrixAgent:
    def __init__(self, papers_dir="/app/papers", output_dir="/app/output"):
        self.papers_dir = papers_dir
        self.output_dir = output_dir

        self.embeddings = get_embeddings()
        print(f"[Agent] Menggunakan LLM provider: {config.LLM_PROVIDER}")
        self.llm = build_llm()

    def process_paper(self, pdf_path):
        filename = os.path.basename(pdf_path)
        print(f"[Agent] Memproses berkas: {filename}")

        pages = load_pdf(pdf_path)
        chunks = split_documents(pages)
        vector_store = build_index(chunks, self.embeddings)
        retriever = create_hybrid_retriever(chunks, vector_store)

        context = retrieve_context(retriever, GAP_MATRIX_QUERY)
        summary = generate(self.llm, context)
        return heal(self.llm, filename, summary, retriever)

    def run(self):
        pdf_files = glob.glob(os.path.join(self.papers_dir, "*.pdf"))
        if not pdf_files:
            print("[Agent] Berkas PDF tidak ditemukan pada direktori input.")
            return

        results = []
        for pdf in pdf_files:
            summary = self.process_paper(pdf)
            results.append(f"### {os.path.basename(pdf)}\n{summary}\n")

        out_file = os.path.join(self.output_dir, "gap_matrix_result.md")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n---\n".join(results))
        print(f"[Agent Selesai] Hasil matriks berhasil digenerate di {out_file}")
