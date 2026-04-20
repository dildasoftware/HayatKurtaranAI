"""
backend/vector_db.py
====================
Antigravity RAG Mimarisi - Vektör Veritabanı Modülü

Bu modül:
  1. data/ klasöründeki .txt dosyalarını yükler
  2. Metinleri anlamlı parçalara (semantic chunks) böler
  3. sentence-transformers ile embedding oluşturur
  4. FAISS üzerinde vektör indeksi kurar
  5. get_context(query) ile ilgili chunk'ları döndürür
"""

import os
import re
import numpy as np  # type: ignore
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from backend.config import (  # pyre-ignore
    VECTOR_MODEL_NAME,
    VECTOR_TOP_K,
    VECTOR_SIMILARITY_THRESHOLD,
    CRITICAL_KEYWORDS,
)

# ─── Konfigürasyon ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ─── Global Değişkenler (lazy-load ile doldurulur) ─────────────────────────────
_model: SentenceTransformer = None
_index: faiss.IndexFlatIP = None
_chunks: list[dict] = []  # {"text": str, "source": str, "metadata": dict}


def _load_model() -> SentenceTransformer:
    """Modeli bir kez yükler, sonraki çağrılarda önbellekten döner."""
    global _model
    if _model is None:
        print("[VectorDB] Embedding modeli yükleniyor...")
        _model = SentenceTransformer(VECTOR_MODEL_NAME)
        print("[VectorDB] Model hazır.")
    return _model


def _load_and_chunk_data() -> list[dict]:
    """
    data/ klasöründeki .txt dosyalarını yükler,
    başlıkları ve içeriklerini anlamlı şekilde birleştirir.
    """
    chunks = []
    if not os.path.exists(DATA_DIR):
        print(f"[VectorDB] UYARI: Veri dizini bulunamadı: {DATA_DIR}")
        return chunks

    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not txt_files:
        print("[VectorDB] UYARI: data/ klasöründe .txt dosyası bulunamadı.")
        return chunks

    for filename in txt_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            raw_chunks = re.split(r'\n{2,}', content.strip())
            
            merged_chunks = []
            current_title = ""
            for chunk in raw_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Başlıkları yakalamak için (genelde tek satır ve kısa olur)
                if '\n' not in chunk and len(chunk) < 100:
                    current_title = chunk
                else:
                    if current_title:
                        merged_chunks.append(current_title + "\n" + chunk)
                        current_title = ""
                    else:
                        merged_chunks.append(chunk)

            # Sonda kalan başlık varsa
            if current_title:
                merged_chunks.append(current_title)

            for i, chunk_text in enumerate(merged_chunks):
                if len(chunk_text) < 30:  # Çok kısa parçaları atla
                    continue

                # İlk satır genellikle başlıktır
                first_line = chunk_text.split('\n')[0].strip()
                chunks.append({
                    "text": chunk_text,
                    "source": filename.replace(".txt", ""),
                    "title": first_line,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                    }
                })
        except Exception as e:
            print(f"[VectorDB] Dosya okunamadı: {filename} – {e}")

    print(f"[VectorDB] Toplam {len(chunks)} chunk yüklendi.")
    return chunks


def _build_index(chunks: list[dict]) -> faiss.IndexFlatIP:
    """
    Chunk metinlerini embed eder, normalize eder ve
    FAISS Inner Product (cosine similarity) indeksi kurar.
    """
    model = _load_model()
    texts = [c["text"] for c in chunks]
    print(f"[VectorDB] {len(texts)} chunk için embedding oluşturuluyor...")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine Similarity (normalize edilmiş vektörler için)
    index.add(embeddings)
    print(f"[VectorDB] FAISS indeksi kuruldu. Vektör boyutu: {dim}")
    return index


def initialize():
    """
    Vektör veritabanını ilk kez veya yeniden başlatır.
    Uygulama başlangıcında bir kez çağrılmalıdır.
    """
    global _chunks, _index
    _chunks = _load_and_chunk_data()
    if _chunks:
        _index = _build_index(_chunks)
    else:
        print("[VectorDB] UYARI: Chunk bulunamadı, indeks oluşturulamadı.")


def is_critical_query(query: str) -> bool:
    """
    Kullanıcının sorgusunda hayati tehlike içeren anahtar kelime var mı kontrol eder.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in CRITICAL_KEYWORDS)


def _enhance_query(query: str) -> str:
    """
    Kullanıcının yazdığı günlük dildeki veya hatalı kelimeleri (yaktım, böcek sokması, araba çarptı)
    sistemdeki teknik karşılıklarına (yanık, anafilaksi, travma) bağlamak için ek anahtar kelimeler üretir.
    Cümle embedding kalitesini ve FAISS sonucunu katlayarak arttırır.
    """
    ql = query.lower().replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    ek = []
    
    if "yandi" in ql or "yakti" in ql or "yanik" in ql:
        ek.append("yanık yanma termal isi hasarı")
    if "kesti" in ql or "kaniyo" in ql or "kanama" in ql or "kanar" in ql:
        ek.append("kesik kanama açık yara")
    if "bocek" in ql or "ari " in ql or "soktu" in ql or "isirdi" in ql or "siyan" in ql or "yilan" in ql or "akrep" in ql:
        ek.append("ısırık sokma alerji anafilaksi hayvan zehir")
    if "nefes" in ql or "bogul" in ql or "tikandi" in ql or "yutam" in ql:
        ek.append("boğulma heimlich astım hava yolu")
    if "basim don" in ql or "bayil" in ql or "gozum karar" in ql or "bilinc" in ql:
        ek.append("bayılma senkop vertigo bilinç kaybı şok pozisyonu")
    if "carp" in ql or "dustu" in ql or "kirildi" in ql or "kaza" in ql:
        ek.append("travma kırık incinme atel kafa kanaması")
    if "zehir" in ql or "ilac icti" in ql or "mide" in ql or "kustu" in ql:
        ek.append("zehirlenme kusma bulantı 114")
    if "seker" in ql or "diyabet" in ql or "hipogli" in ql:
        ek.append("şeker düşmesi hipoglisemi diyabetik koma")
    if "tansiyon" in ql or "bas agrisi" in ql:
        ek.append("hipertansiyon baş ağrısı kan basıncı")
    if "cocuk" in ql or "bebek" in ql or "yeni dogan" in ql:
        ek.append("bebek çocuk pediatrik acil durum")
    if "hamile" in ql or "gebe" in ql or "hamilelik" in ql:
        ek.append("gebelik hamilelik riskli gebelik acil durum")
        
    if ek:
        # Eski sorgunun ardına zenginleştirilmiş kelimeleri ekleyerek 
        # modelin doğru chunk'ı bulmasını garantile
        return query + " " + " ".join(ek)
    return query


def get_context(query: str) -> tuple[str, list[dict], bool]:
    """
    Kullanıcı sorgusuna en alakalı metin parçalarını döndürür.

    Returns:
        context_text (str): LLM'e gönderilecek birleşik bağlam metni
        sources (list[dict]): Kaynak bilgileri (kaynak gösterimi için)
        is_critical (bool): Sorgu kritik/acil durumu tetikliyor mu?
    """
    global _chunks, _index

    # Sistem hazır değilse başlat
    if not _chunks or _index is None:
        initialize()

    if not _chunks or _index is None:
        return "", [], is_critical_query(query)

    is_critical = is_critical_query(query)

    # Sorguyu zenginleştir (Keyword/Synonym bazlı)
    enhanced_query = _enhance_query(query)

    # Sorguyu embed et
    model = _load_model()
    query_embedding = model.encode([enhanced_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # FAISS arama
    actual_k = min(VECTOR_TOP_K, len(_chunks))
    assert _index is not None
    scores, indices = _index.search(query_embedding, actual_k)

    results = []
    # Pyre2 tip engelini asmak icin:
    # pyre-ignore
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue

        # Benzerlik eşiği kontrolü
        if score < VECTOR_SIMILARITY_THRESHOLD:
            continue

        chunk = _chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        results.append(chunk)

    if not results:
        return "", [], is_critical

    # Bağlam metnini oluştur
    context_parts = []
    sources = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[Kaynak {i}: {r['source']} - {r['title']}]\n{r['text']}")
        sources.append({
            "title": r["title"],
            "source": r["source"],
            "score": r["similarity_score"]
        })

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources, is_critical


# Modül doğrudan çalıştırıldığında test amaçlı kontrol
if __name__ == "__main__":
    initialize()
    test_queries = [
        "Arı soktu ne yapmalıyım?",
        "Kalp durmuş ne yapacağım?",
        "Kafadan düştü ne yapayım?",
        "Nasıl kilo veririm?"
    ]
    for q in test_queries:
        ctx, srcs, crit = get_context(q)
        print(f"\n{'='*60}")
        print(f"Sorgu: {q}")
        print(f"Kritik: {crit}")
        print(f"Kaynaklar: {[s['title'] for s in srcs]}")
        ctx_str = str(ctx)
        # pyre-ignore
        print(f"Bağlam: {ctx_str[:200]}...")
