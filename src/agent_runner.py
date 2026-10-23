"""AI Agent untuk Otomasi Ekstraksi Literature GAP Matrix
Infrastruktur: sentence-transformers (Lokal) + LLM (LM Studio lokal / Groq API)
"""
import os

from rag_pipeline.pipeline import GapMatrixAgent

if __name__ == "__main__":
    agent = GapMatrixAgent(
        papers_dir=os.getenv("PAPERS_DIR", "/app/papers"),
        output_dir=os.getenv("OUTPUT_DIR", "/app/output"),
    )
    agent.run()
