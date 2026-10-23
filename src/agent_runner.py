"""
AI Agent untuk Otomasi Ekstraksi Literature GAP Matrix
Infrastruktur: sentence-transformers (Lokal) + LLM (LM Studio lokal / Groq API)
"""
import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").lower()


def build_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://192.168.200.6:1234/v1"),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        model=os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct"),
        temperature=0.1
    )


class GapMatrixAgent:
    def __init__(self, papers_dir="/app/papers", output_dir="/app/output"):
        self.papers_dir = papers_dir
        self.output_dir = output_dir

        print("[Agent] Inisialisasi model embedding lokal (mpnet-base-v2)...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        print(f"[Agent] Menggunakan LLM provider: {LLM_PROVIDER}")
        self.llm = build_llm()

    def process_paper(self, pdf_path):
        filename = os.path.basename(pdf_path)
        print(f"[Agent] Memproses berkas: {filename}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)

        db = FAISS.from_documents(chunks, self.embedding)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        query = "Ekstrak Problem, Metode, Dataset, Temuan, Keterbatasan, dan Future Work."
        relevant_chunks = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_chunks])

        prompt = f"""
        Berdasarkan konteks artikel ilmiah terlampir, ekstrak informasi untuk Literature GAP Matrix:
        Konteks:
        {context}

        Format output WAJIB:
        Problem: <uraian ringkas>
        Metode: <uraian arsitektur & model>
        Dataset: <nama dataset / korpus>
        Temuan: <hasil eksperimen utama>
        Keterbatasan: <limitasi sistem>
        Future Work: <arah riset masa depan>
        """

        response = self.llm.invoke(prompt)
        return self._evaluate_and_heal(filename, response.content, retriever)

    def _evaluate_and_heal(self, filename, text_result, retriever):
        """Mekanisme verifikasi mandiri agen jika atribut belum lengkap"""
        required = ["Problem:", "Metode:", "Dataset:", "Temuan:", "Keterbatasan:", "Future Work:"]
        missing = [f for f in required if f not in text_result]

        if missing:
            print(f"[Agent Warning] Atribut {missing} hilang pada {filename}. Melakukan perbaikan konteks...")
            patch_query = f"Cari informasi eksplisit mengenai: {', '.join(missing)}"
            extra_ctx = "\n".join([d.page_content for d in retriever.invoke(patch_query)])

            repair_prompt = f"""
            Lengkapi atribut {missing} berdasarkan konteks tambahan berikut:
            {extra_ctx}

            Hasil sebelumnya:
            {text_result}
            """
            repaired = self.llm.invoke(repair_prompt)
            return repaired.content

        return text_result

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


if __name__ == "__main__":
    agent = GapMatrixAgent(
        papers_dir=os.getenv("PAPERS_DIR", "/app/papers"),
        output_dir=os.getenv("OUTPUT_DIR", "/app/output"),
    )
    agent.run()
