# LAPORAN TUGAS AWAL RTM 2: LITERATURE GAP MATRIX & PIPELINE RAG
**Otomasi Ekstraksi Gap Riset Menggunakan Sentence-Transformers Lokal dan Groq API**
*Studi Kasus: Sistem Kepatuhan Regulasi Fintech & Perlindungan Data Pribadi (ReguLaw RAG)*

---

| Atribut Dokumen | Keterangan |
| :--- | :--- |
| **Mata Kuliah** | Introduction to Generative AI (20261-P36102) |
| **Program Studi** | Magister Artificial Intelligent (S2) |
| **Institusi** | Universitas Dian Nuswantoro, Semarang |
| **Kode Kelas** | P3612 |
| **Disusun Oleh** | 1. Rofiq Fauzi – P36.2026.00047<br>2. Hasan Shobri – P36.2026.00004 |
| **Dosen Pengampu** | 1. Prof. Ir. Heru Agus Santoso, Ph.D., IPM., ASEAN Eng<br>2. Dr. Eng. Farrikh Alzami, M.Kom |
| **Tanggal** | 28 September 2026 |
| **URL Repositori GitHub** | [https://github.com/rofiqfauzi/regulaw-rag-gap-matrix](https://github.com/rofiqfauzi/regulaw-rag-gap-matrix) |

---

## 1. Pendahuluan & Latar Belakang

Model Bahasa Besar (*Large Language Model* / LLM) menawarkan kemampuan penalaran bahasa alami yang tinggi. Namun, implementasi model generatif secara murni (*stand-alone zero-shot*) dihadapkan pada dua kendala fundamental:
1. **Halusinasi Faktual (*Hallucination*):** Fenomena ketika model mengarang jawaban yang salah namun disajikan dengan tingkat keyakinan tinggi. Pada domain hukum (*legal hallucination*), kesalahan mengutip nomor pasal atau klausul sanksi bersifat fatal.
2. **Keterbatasan Jendela Konteks (*Context Window*) & Biaya:** Menyertakan ratusan halaman dokumen tebal pada setiap pemanggilan API menghasilkan latensi tinggi dan konsumsi token yang sangat mahal.

Arsitektur **Retrieval-Augmented Generation (RAG)** menyelesaikan masalah ini dengan memisahkan basis pengetahuan eksternal dari bobot parameter model. Jawaban yang dihasilkan bersifat *strictly grounded* pada dokumen otoritatif yang berhasil ditarik (*retrieved*).

Dalam proyek **ReguLaw RAG**, sistem difokuskan pada otomasi verifikasi kepatuhan regulasi finansial teknologi (fintech) di Indonesia, dengan korpus utama:
* **UU No. 27 Tahun 2022** tentang Pelindungan Data Pribadi (UU PDP).
* **POJK No. 10/POJK.05/2022** tentang Layanan Pendanaan Bersama Berbasis Teknologi Informasi.
* **Surat Edaran OJK (SEOJK)** terkait Manajemen Risiko Teknologi Informasi.

Laporan ini memuat perancangan pipeline RAG hibrida, penggelaran sistem pada lingkungan Ubuntu dan Docker berbantuan AI Agent, serta ekstraksi otomatis **Literature GAP Matrix** dari 10 artikel ilmiah internasional bereputasi.

---

## 2. Arsitektur Pipeline RAG & Lingkungan Deployment

Sesuai materi perkuliahan Bab 5, pipeline dibangun di atas 6 tahap baku:
1. **Document Processing:** Ekstraksi teks dari berkas PDF dokumen ilmiah.
2. **Chunking:** Pemotongan dokumen menjadi unit teks semantik yang koheren.
3. **Embedding Lokal:** Menggunakan model representasi lokal `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dimensi) guna menjamin privasi dokumen dan kemandirian dari API cloud komersial.
4. **Penyimpanan Vektor:** Pengindeksan vektor secara efisien menggunakan FAISS (*Facebook AI Similarity Search*).
5. **Retrieval:** Pengambilan *top-k* potongan teks yang memiliki kemiripan semantik tertinggi dengan kueri.
6. **Augmented Generation:** Penyusunan prompt dengan instruksi ketat (*strict grounding*) yang dieksekusi oleh model `llama-3.3-70b-versatile` melalui Groq API berkecepatan tinggi.

```
+-----------------------------------------------------------------------------------+
|                         INGESTION & INDEXING (LOCAL OFFLINE)                      |
|                                                                                   |
|  [PDF Papers / Docs] ---> [PyPDF Extraction] ---> [Recursive Text Splitter]        |
|                                                                 │                 |
|                                                                 ▼                 |
|  [FAISS Vector Store] <--- [Paraphrase-Multilingual-MPNet] <───┘                 |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (Semantic Similarity Search: Top-k)
+-----------------------------------------------------------------------------------+
|                       AGENTIC RETRIEVAL & GENERATION (GROQ API)                   |
|                                                                                   |
|  [User/Matrix Query] ---> [Retriever k=5] ---> [Strict Grounded System Prompt]    |
|                                                                 │                 |
|                                                                 ▼                 |
|  [Literature GAP Matrix Table] <--- [Self-Healing Agent] <─── [Groq: Llama-3.3]   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Konfigurasi Sistem (Ubuntu + Docker + AI Agent)

Sistem dirancang untuk berjalan secara terisolasi dan portabel di lingkungan server atau desktop Ubuntu menggunakan Docker.

### 3.1 `requirements.txt`
```text
langchain==0.3.1
langchain-community==0.3.1
langchain-groq==0.2.0
langchain-text-splitters==0.3.0
sentence-transformers==3.1.1
faiss-cpu==1.8.0.post1
pypdf==5.0.1
python-dotenv==1.0.1
pydantic==2.9.2
```

### 3.2 `Dockerfile`
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/agent_runner.py"]
```

### 3.3 `docker-compose.yml`
```yaml
version: '3.8'

services:
  rag-agent:
    build: .
    container_name: regulaw_rag_agent
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_HOME=/app/cache/huggingface
    volumes:
      - ./papers:/app/papers
      - ./output:/app/output
      - ./model_cache:/app/cache
    restart: unless-stopped
```

### 3.4 Skrip Python AI Agent (`src/agent_runner.py`)
```python
"""
AI Agent untuk Otomasi Ekstraksi Literature GAP Matrix
Infrastruktur: sentence-transformers (Lokal) + Groq Cloud API
"""
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GapMatrixAgent:
    def __init__(self, papers_dir="/app/papers", output_dir="/app/output"):
        self.papers_dir = papers_dir
        self.output_dir = output_dir

        print("[Agent] Inisialisasi model embedding lokal (mpnet-base-v2)...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

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
    agent = GapMatrixAgent()
    agent.run()
```

---

## 4. Literature GAP Matrix (10 Artikel Ilmiah)

Berikut adalah tabel matriks perbandingan terhadap sepuluh artikel ilmiah bertaraf internasional pada domain RAG hukum, audit kepatuhan finansial, dan regulasi privasi data:

| Paper | DOI | Problem | Metode | Dataset | Temuan | Keterbatasan | Future Work |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[Paper 1]** Rahman et al. (2025) - *LQ-RAG* | `10.1109/ACCESS.2025.3542125` (IEEE Access) | Halusinasi hukum mencapai 58–82% pada naive RAG saat merujuk pasal spesifik. | LQ-RAG: Dual fine-tuning (embedding & generative), multi-agent prompt, feedback loop otomatis. | Korean Legal QA & Statutory statutes corpus. | Mengurangi halusinasi secara signifikan; akurasi penarikan klausul meningkat >24% dibanding baseline. | Ketergantungan tinggi pada model evaluator komersial (proprietary LLM API) yang mahal. | Membangun agen evaluator domain-spesifik berbasis model open-weight lokal on-premise. |
| **[Paper 2]** Li et al. (2026) - *LegRAG* | `10.48550/arXiv.2603.11772` (cs.CL / Review) | Pemotongan token acak (*naive chunking*) merusak hierarki klausa perundang-undangan (*statutory loss*). | LegRAG: Clause-boundary chunking, dual-path self-reflection verification, Qwen3-Max evaluator. | Legal-DC benchmark (10.000+ pasal hukum publik & perdata). | Mengungguli SOTA RAG sebesar 1.3%–5.6% pada metrik akurasi yuridis dan kelengkapan pasal. | Biaya komputasi inferensi self-reflection tinggi untuk melayani kueri interaktif real-time. | Optimasi decoding spekulatif dan adaptasi korpus bahasa hukum lintas yurisdiksi. |
| **[Paper 3]** Cosentino et al. (2026) - *Auditable RAG* | `10.3390/fi18060284` (MDPI Future Internet) | Sistem RAG finansial tidak memiliki *audit trail* (jejak bukti) yang dapat diverifikasi regulator perbankan. | Provenance-aware RAG: Hybrid indexing (Dense + Sparse BM25 + ColBERT) + JSON schema citation. | Laporan Keuangan 10-K SEC & Peraturan Perbankan IFRS/Basel. | Grounding score mencapai 94.2%; klaim kepatuhan dapat dilacak balik presisi hingga nomor token teks. | Belum menguji regulasi privasi data dinamis yang berpotensi memuat kontradiksi silang. | Otomasi deteksi kepatuhan dinamis saat terjadi amandemen regulasi secara instan. |
| **[Paper 4]** Sha et al. (2026) - *Rhetorical RAG* | `10.48550/arXiv.2608.06828` (cs.IR) | Perbedaan struktur teks putusan pengadilan membuat pencarian kemiripan kosinus biasa menghasilkan noise. | Rhetorical-Role segmentation (Fakta, Isu, Putusan) + Hierarchical dense retriever. | Putusan Mahkamah Agung India (2.500 berkas putusan resmi). | Contextual recall meningkat 18.5% dibanding dense search flat konvensional. | Hanya diuji pada preseden peradilan (*case law*), belum pada undang-undang tertulis (*statutory law*). | Mengintegrasikan kodifikasi hukum perundang-undangan dengan interpretasi hakim. |
| **[Paper 5]** Bankura et al. (2026) - *LegalAI Assistant* | `10.1109/NEXO-TECH62341.2026.10823411` (IEEE Nexo-Tech) | Isu privasi data: kantor hukum dilarang mengirimkan dokumen klien ke API cloud LLM pihak ketiga. | 100% On-premise RAG: Local SLM (Llama-3-8B), quantized embeddings, FAISS vector store lokal. | Korpus 21 Undang-Undang Primer Nasional & Regulasi Bisnis. | Hit rate retrieval 82.1%, latency 1.4 detik, dan menjamin nol kebocoran data keluar jaringan. | Kapasitas penalaran (*reasoning*) SLM lokal terbatas saat memproses skenario multi-pasal kompleks. | Implementasi *speculative agentic retrieval* untuk mendongkrak reasoning SLM lokal. |
| **[Paper 6]** Waikar et al. (2025) - *RegAudit RAG* | `10.1111/cts.13924` (Clinical & Transl. Sci.) | Tingginya biaya dan risiko audit manual terhadap kepatuhan SOP akibat regulasi yang sering berganti. | Semantic parsing berbasis checklist kepatuhan formal + LLM strict factual grounding. | Regulasi FDA & Dokumen Panduan Protokol Klinis. | Menghemat 60% waktu review dokumen manual dengan akurasi verifikasi klausul mencapai 91%. | Format tabel multi-kolom dan diagram alir sering gagal diekstrak oleh parser PDF standar. | Integrasi Vision-Language Model (VLM) untuk dokumen regulasi dengan tata letak visual kompleks. |
| **[Paper 7]** Sapsinchai & Nonsiri (2025) | `10.1109/ICSEC63512.2025.10845321` (IEEE ICSEC) | Pengguna awam kesulitan memformulasikan kueri bahasa hukum formal saat berkonsultasi. | Query Rewriter Agent + Hierarchical Document Indexing + Feedback-driven prompt adaptation. | UU Perlindungan Konsumen & Klausul Standar Baku Kontrak. | Meningkatkan relevansi penarikan dokumen sebesar 28% dari pertanyaan masyarakat umum yang ambigu. | Kurang optimal dalam menangani dialek atau istilah hukum daerah informal di luar bahasa standar. | Fine-tuning model representasi multilingual dengan korpus percakapan lokal interaktif. |
| **[Paper 8]** Dahl et al. (2024) - *Legal Hallucination* | `10.1111/jels.12384` (Wiley JELS) | LLM komersial sering melakukan halusinasi sitasi (mengarang nomor bab, pasal, dan tahun regulasi). | Studi empiris komparatif: Evaluasi zero-shot LLM vs Standard RAG pada 5 benchmark hukum formal. | Stanford Legal Benchmark Corpus (Pertanyaan Hukum Terstandar). | RAG memangkas halusinasi sitasi dari 69% menjadi di bawah 7%, namun masih menyisakan unsupported extrapolation. | Murni studi benchmarking diagnostik; tidak mengusulkan modul mitigasi arsitektur baru. | Pengembangan *constraint decoding* terintegrasi agar LLM menolak merespons tanpa bukti pasal rujukan. |
| **[Paper 9]** Kwon et al. (2024) - *CBR-RAG* | `10.1007/978-3-031-78901-2_16` (Springer LNAI) | Aturan hukum tertulis tidak dapat berdiri sendiri tanpa interpretasi kontekstual kasus konkret sebelumnya. | Integrasi siklus Case-Based Reasoning (CBR: Retrieve, Reuse, Revise, Retain) dengan Dense Vector Search. | Korpus Hukum Perdata & Putusan Sengketa Kontrak Bisnis. | Menghasilkan penalaran analogis yang lebih kokoh bagi praktisi hukum dibanding RAG standar. | Pengindeksan kasus lama membutuhkan anotasi pakar hukum manual yang mahal dan lambat. | Otomatisasi ekstraksi fitur kasus menggunakan model zero-shot information extraction. |
| **[Paper 10]** Zhang et al. (2024) - *FinAudit-RAG* | `10.1109/TCSS.2024.3391820` (IEEE Trans. CSS) | Inkonsistensi verifikasi antara laporan teks deskriptif (*disclosure*) dan batas angka kepatuhan finansial. | Hybrid Multi-Vector Retriever + Rule-Engine Symbolic Filter terintegrasi. | Laporan Tahunan Korporasi Finansial & Aturan Rasio Keuangan SEC/POJK. | Menghilangkan salah interpretasi angka neraca terhadap ketentuan batas modal minimum regulasi. | Skalabilitas aturan simbolik terbatas saat jumlah variabel kepatuhan berkembang menjadi ribuan. | Pengembangan Graph-RAG (Knowledge Graph) berbasis ontologi regulasi yang adaptif. |

---

## 5. Analisis Sintesis Research Gap

Berdasarkan sintesis sepuluh literatur ilmiah pada matriks di atas, teridentifikasi tiga celah riset fundamental (*research gaps*) yang melandasi proyek ini:

1. **Celah Integritas Semantik Yuridis (*Semantic Integrity Gap*):** Sebagian besar penelitian terdahulu (seperti Dahl et al., 2024; Bankura et al., 2026) masih mengandalkan pemotongan teks berbasis jendela token arbitrer (*fixed token chunking*, misal 500–1000 token). Pada dokumen undang-undang tertulis (*statutory law*), metode ini memutus relasi hierarkis antara Bab, Bagian, Pasal, Ayat, dan Penjelasan Pasal. Ketika sebuah pasal terpotong di tengah kalimat, model kehilangan konteks klausul pengecualian (*subject to conditions*) yang terkandung pada ayat berikutnya. Meskipun Li et al. (2026) mengusulkan *clause-boundary chunking*, pendekatan tersebut belum mengakomodasi ketergantungan relasional antar-pasal yang saling mengikat (misalnya keterkaitan definisi pada Bab Ketentuan Umum terhadap klausul sanksi pada Bab Penutup).
2. **Trade-Off Auditabilitas vs. Privasi Data Lokal (*Auditability-Privacy Trade-off*):** Penelitian dengan tingkat keterlacakan tinggi (Cosentino et al., 2026; Rahman et al., 2025) sangat bergantung pada API model komersial tertutup (*proprietary cloud LLM* seperti GPT-4). Hal ini menimbulkan kendala kepatuhan perbankan karena SOP internal dan data rahasia tidak boleh dikirimkan ke cloud pihak ketiga. Di sisi lain, penelitian yang berfokus pada *on-premise deployment* (Bankura et al., 2026) belum dilengkapi mekanisme rekayasa jejak bukti (*provenance audit trail*) terstruktur berbasis JSON yang mampu membuktikan validitas rujukan hingga nomor token dan metadata ayat.
3. **Celah Rekonsiliasi Regulasi Finansial Indonesia (*Regulatory Cross-Referencing Gap*):** Belum ada arsitektur RAG yang dirancang khusus untuk memitigasi tumpang tindih kewajiban hukum antara undang-undang payung privasi data (UU No. 27/2022 tentang Pelindungan Data Pribadi) dan regulasi operasional sektoral fintech (POJK No. 10/POJK.05/2022). Karakteristik bahasa hukum Indonesia memerlukan mekanisme validasi silang (*cross-regulatory self-reflection*) agar sistem mampu membedakan sanksi administratif sektoral OJK dengan ancaman pidana umum UU PDP.

---

## 6. Usulan Paper Ilmiah untuk Publikasi Kelompok

### 6.1 Judul Paper yang Diusulkan
> **"Hierarchical Clause-Aware RAG with Strict Provenance Auditing for Fintech Regulatory Compliance under Indonesian Privacy Laws"**
> *(Sistem RAG Berbasis Chunking Hierarki Pasal dengan Audit Jejak Bukti Ketat untuk Verifikasi Kepatuhan Regulasi Fintech Berdasarkan UU PDP dan POJK)*

### 6.2 Kontribusi Kebaruan Riset (*Novelty*)
* **Novelty 1 – Hierarchical Clause-Aware Chunking (HCAC):** Membagi korpus dokumen hukum mengikuti struktur pohon yuridis resmi (*Undang-Undang → Bab → Bagian → Pasal → Ayat*), mempertahankan relasi induk-anak (*parent-child metadata*) agar klausul pengecualian hukum tidak terdistorsi saat di-retrieve.
* **Novelty 2 – Dual-Layer On-Premise Auditability Engine:** Mengintegrasikan model representasi lokal `paraphrase-multilingual-mpnet-base-v2` dengan model inferensi lokal yang menghasilkan format keluaran terstruktur (JSON). Setiap kesimpulan hukum memuat kutipan teks *verbatim*, nomor pasal rujukan, dan skor keyakinan (*confidence score*) guna memenuhi standar audit industri perbankan.
* **Novelty 3 – Cross-Regulatory Self-Reflection:** Menerapkan agen refleksi ganda yang secara otomatis memvalidasi silang draf SOP internal terhadap UU PDP dan POJK No. 10/2022 sekaligus, sehingga potensi benturan atau ketidakpatuhan hukum terdeteksi secara otomatis sebelum laporan diserahkan kepada auditor.
