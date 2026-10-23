"""Self-Healing Agent: verifikasi kelengkapan atribut matriks dan perbaikan otomatis."""
from .retrieval import retrieve_context

REQUIRED_ATTRIBUTES = ["Problem:", "Metode:", "Dataset:", "Temuan:", "Keterbatasan:", "Future Work:"]


def find_missing(text_result):
    return [attr for attr in REQUIRED_ATTRIBUTES if attr not in text_result]


def heal(llm, filename, text_result, retriever):
    missing = find_missing(text_result)
    if not missing:
        return text_result

    print(f"[Agent Warning] Atribut {missing} hilang pada {filename}. Melakukan perbaikan konteks...")
    patch_query = f"Cari informasi eksplisit mengenai: {', '.join(missing)}"
    extra_ctx = retrieve_context(retriever, patch_query)

    repair_prompt = f"""
    Lengkapi atribut {missing} berdasarkan konteks tambahan berikut:
    {extra_ctx}

    Hasil sebelumnya:
    {text_result}
    """
    return llm.invoke(repair_prompt).content
