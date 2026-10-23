"""Web GUI untuk mengetes pipeline RAG ReguLaw (development server)."""
import glob
import os
import time

from flask import Flask, jsonify, request, send_from_directory

from rag_pipeline import config
from rag_pipeline.chunking import split_documents
from rag_pipeline.document_processing import load_pdf
from rag_pipeline.embedding import get_embeddings
from rag_pipeline.generation import GAP_MATRIX_QUERY, build_llm, generate
from rag_pipeline.retrieval import create_retriever
from rag_pipeline.vector_store import build_index

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS_DIR = os.getenv("PAPERS_DIR", os.path.join(BASE_DIR, "papers"))

app = Flask(__name__, static_folder="static", static_url_path="/static")

STATE = {
    "embeddings": None,
    "llm": None,
    "pdf": None,
    "retriever": None,
    "chunk_count": 0,
    "indexed_at": None,
}


def get_embeddings_lazy():
    if STATE["embeddings"] is None:
        STATE["embeddings"] = get_embeddings()
    return STATE["embeddings"]


def get_llm_lazy():
    if STATE["llm"] is None:
        STATE["llm"] = build_llm()
    return STATE["llm"]


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/info")
def info():
    lmstudio = config.LLM_PROVIDER == "lmstudio"
    return jsonify({
        "provider": config.LLM_PROVIDER,
        "model": config.LM_STUDIO_MODEL if lmstudio else config.GROQ_MODEL,
        "base_url": config.LM_STUDIO_BASE_URL if lmstudio else "https://api.groq.com",
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "indexed_pdf": STATE["pdf"],
        "chunk_count": STATE["chunk_count"],
    })


@app.get("/api/papers")
def papers():
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PAPERS_DIR, "*.pdf")))
    return jsonify({"papers": files})


@app.post("/api/index")
def index_pdf():
    filename = request.form.get("filename")
    upload = request.files.get("file")
    if upload is not None and upload.filename:
        filename = upload.filename
        upload.save(os.path.join(PAPERS_DIR, filename))
    if not filename:
        return jsonify({"error": "Field 'filename' atau file upload wajib diisi."}), 400
    path = os.path.join(PAPERS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": f"Berkas tidak ditemukan: {filename}"}), 404

    started = time.time()
    pages = load_pdf(path)
    chunks = split_documents(pages)
    vector_store = build_index(chunks, get_embeddings_lazy())
    STATE.update({
        "pdf": filename,
        "retriever": create_retriever(vector_store),
        "chunk_count": len(chunks),
        "indexed_at": time.strftime("%H:%M:%S"),
    })
    return jsonify({
        "pdf": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "seconds": round(time.time() - started, 2),
    })


def require_index():
    if STATE["retriever"] is None:
        return jsonify({"error": "Indeks belum dibangun. Index dokumen terlebih dahulu."}), 400
    return None


def docs_to_json(docs):
    return [
        {"rank": i + 1, "page": doc.metadata.get("page"), "content": doc.page_content}
        for i, doc in enumerate(docs)
    ]


@app.post("/api/query")
def query():
    error = require_index()
    if error:
        return error
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Pertanyaan kosong."}), 400

    started = time.time()
    docs = STATE["retriever"].invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
    Jawab pertanyaan berdasarkan KONTEKS berikut. Jawab hanya menggunakan informasi
    yang ada di konteks; jika jawabannya tidak ada di konteks, katakan tidak ditemukan.
    Jawab dalam bahasa Indonesia.

    KONTEKS:
    {context}

    PERTANYAAN: {question}
    """
    answer = get_llm_lazy().invoke(prompt).content
    return jsonify({
        "pdf": STATE["pdf"],
        "question": question,
        "answer": answer,
        "seconds": round(time.time() - started, 2),
        "retrieved": docs_to_json(docs),
    })


@app.post("/api/gap_matrix")
def gap_matrix():
    error = require_index()
    if error:
        return error

    started = time.time()
    docs = STATE["retriever"].invoke(GAP_MATRIX_QUERY)
    context = "\n\n".join(doc.page_content for doc in docs)
    result = generate(get_llm_lazy(), context)
    return jsonify({
        "pdf": STATE["pdf"],
        "query": GAP_MATRIX_QUERY,
        "result": result,
        "seconds": round(time.time() - started, 2),
        "retrieved": docs_to_json(docs),
    })


if __name__ == "__main__":
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "5000"))
    print(f"[Web] ReguLaw RAG Tester di http://{host}:{port}")
    app.run(host=host, port=port)
