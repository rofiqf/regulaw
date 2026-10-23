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
from rag_pipeline.retrieval import create_hybrid_retriever
from rag_pipeline.vector_store import build_index

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS_DIR = os.getenv("PAPERS_DIR", os.path.join(BASE_DIR, "papers"))
ENV_PATH = os.path.join(BASE_DIR, ".env")

app = Flask(__name__, static_folder="static", static_url_path="/static")

STATE = {
    "embeddings": None,
    "llm": None,
    "pdf": None,
    "chunks": None,
    "vector_store": None,
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
    models = {
        "lmstudio": config.LM_STUDIO_MODEL,
        "gemini": config.GEMINI_MODEL,
        "groq": config.GROQ_MODEL,
    }
    return jsonify({
        "provider": config.LLM_PROVIDER,
        "model": models.get(config.LLM_PROVIDER, ""),
        "base_url": (config.LM_STUDIO_BASE_URL if config.LLM_PROVIDER == "lmstudio"
                     else "https://generativelanguage.googleapis.com" if config.LLM_PROVIDER == "gemini"
                     else "https://api.groq.com"),
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "indexed_pdf": STATE["pdf"],
        "chunk_count": STATE["chunk_count"],
    })


PROVIDER_ENV = {
    "lmstudio": ("LM_STUDIO_MODEL", "LM_STUDIO_API_KEY"),
    "gemini": ("GEMINI_MODEL", "GEMINI_API_KEY"),
    "groq": ("GROQ_MODEL", "GROQ_API_KEY"),
}


def mask_key(key):
    if not key:
        return ""
    return "*" * (len(key) - 4) + key[-4:] if len(key) > 4 else "****"


def persist_env(updates):
    lines, seen = [], set()
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    name = stripped.split("=", 1)[0].strip()
                    if name in updates:
                        lines.append(f"{name}={updates[name]}\n")
                        seen.add(name)
                        continue
                lines.append(line)
    for name, value in updates.items():
        if name not in seen:
            lines.append(f"{name}={value}\n")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def current_settings():
    model_env, key_env = PROVIDER_ENV.get(config.LLM_PROVIDER, PROVIDER_ENV["lmstudio"])
    key = {"GEMINI_API_KEY": config.GEMINI_API_KEY,
           "GROQ_API_KEY": config.GROQ_API_KEY,
           "LM_STUDIO_API_KEY": config.LM_STUDIO_API_KEY}[key_env]
    model = {"LM_STUDIO_MODEL": config.LM_STUDIO_MODEL,
             "GEMINI_MODEL": config.GEMINI_MODEL,
             "GROQ_MODEL": config.GROQ_MODEL}[model_env]
    return {"provider": config.LLM_PROVIDER, "model": model, "api_key_masked": mask_key(key)}


@app.get("/api/settings")
def get_settings():
    return jsonify(current_settings())


@app.post("/api/settings")
def save_settings():
    data = request.get_json(force=True, silent=True) or {}
    provider = (data.get("provider") or "").strip().lower()
    if provider not in PROVIDER_ENV:
        return jsonify({"error": f"Provider tidak dikenal: {provider}"}), 400

    model_env, key_env = PROVIDER_ENV[provider]
    updates = {"LLM_PROVIDER": provider}
    model = (data.get("model") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    if model:
        updates[model_env] = model
    if api_key:
        updates[key_env] = api_key

    persist_env(updates)
    config.configure_llm(provider=provider, model=model or None, api_key=api_key or None)
    STATE["llm"] = None
    return jsonify({"saved": True, **current_settings()})


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
        "chunks": chunks,
        "vector_store": vector_store,
        "retriever": create_hybrid_retriever(chunks, vector_store),
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
    try:
        k = int(data.get("k") or config.TOP_K)
    except (TypeError, ValueError):
        k = config.TOP_K

    started = time.time()
    retriever = create_hybrid_retriever(STATE["chunks"], STATE["vector_store"], k=k)
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
    Jawab pertanyaan berdasarkan KONTEKS berikut. Jawab hanya menggunakan informasi
    yang ada di konteks; jika jawabannya tidak ada di konteks, katakan tidak ditemukan.
    Jawab dalam bahasa Indonesia.

    KONTEKS:
    {context}

    PERTANYAAN: {question}
    """
    try:
        answer = get_llm_lazy().invoke(prompt).content
    except Exception as exc:
        return jsonify({"error": f"LLM ({config.LLM_PROVIDER}) gagal: {exc}"}), 502
    return jsonify({
        "pdf": STATE["pdf"],
        "question": question,
        "answer": answer,
        "k": k,
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
    try:
        result = generate(get_llm_lazy(), context)
    except Exception as exc:
        return jsonify({"error": f"LLM ({config.LLM_PROVIDER}) gagal: {exc}"}), 502
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
