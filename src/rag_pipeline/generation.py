"""Tahap 6 Augmented Generation: prompt strict-grounding dieksekusi LLM (LM Studio lokal / Groq)."""
from . import config

GAP_MATRIX_QUERY = "Ekstrak Problem, Metode, Dataset, Temuan, Keterbatasan, dan Future Work."


def build_llm():
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=0.1
        )
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=0.1
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=config.LM_STUDIO_BASE_URL,
        api_key=config.LM_STUDIO_API_KEY,
        model=config.LM_STUDIO_MODEL,
        temperature=0.1
    )


def build_prompt(context):
    return f"""
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


def generate(llm, context):
    return llm.invoke(build_prompt(context)).content
