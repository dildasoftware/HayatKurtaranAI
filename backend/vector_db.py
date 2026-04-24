# -*- coding: utf-8 -*-
"""
backend/vector_db.py — HayatKurtaran AI Vektör Veritabanı v2.0
================================================================
Modernize edilmiş RAG vektör arama motoru.

Yenilikler (v2.0):
  - Smart Chunker entegrasyonu (section-aware, overlap destekli)
  - FAISS index disk cache (restart'ta yeniden hesaplama yok)
  - Confidence score hesaplama (avg_confidence)
  - Genişletilmiş acil kelime tespiti (40+ ifade)
  - Query enhancement (günlük dil → medikal terim)
"""

import os
import re
import hashlib
import pickle
import numpy as np  # type: ignore
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from backend.chunker import chunk_directory  # type: ignore

# ─── Konfigürasyon ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".faiss_cache")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Türkçe desteği
TOP_K = 5            # En alakalı kaç chunk — 5 yeterli, 20 halüsinasyon yaratır
SIMILARITY_THRESHOLD = 0.40  # Cosine benzerlik alt sınırı — 0.40 altı = alakasız chunk, reddet

# Acil tetikleyici kelimeler - bunlar algılandığında öncelikli uyarı dönülür
CRITICAL_KEYWORDS = [
    # --- Kardiyak ---
    "kalp krizi", "kalp durdu", "kalp durması", "kalbim durdu",
    "göğsümde ağrı", "göğüs ağrısı", "göğsümde baskı",
    "kalbim çok hızlı", "kalbim duracak gibi",
    # --- Solunum ---
    "nefes almıyor", "nefes durdu", "nefes alamıyorum",
    "nefes almak zor", "nefes darlığı", "boğuluyor", "boğulma",
    # --- Bilinç ---
    "bilinç kaybı", "bilincini kaybetti", "bayıldı", "ölüyor",
    "bilinci yok", "bilinci kapandı", "uyanmıyor",
    # --- Kanama ---
    "şiddetli kanama", "çok kan", "kan durmuyor",
    "çok kan kaybediyor", "kan fışkırıyor",
    # --- İnme / Felç ---
    "inme", "felç", "yüzüm düştü", "yüzü düştü",
    "kolum kalkmıyor", "dilim dönmüyor", "konuşamıyor",
    # --- Alerjik / Anafilaksi ---
    "anafilaksi", "alerjik şok", "boğazım şişiyor", "dilim şişiyor",
    # --- Travma ---
    "yüksekten düştü", "kafa travması", "kafasını çarptı",
    "elektrik çarptı", "cereyana kapıldı",
    # --- Zehirlenme ---
    "zehir içti", "ilaç içti", "hapları yuttu",
    # --- Su / Boğulma ---
    "suya düştü", "suda boğulma",
    # --- Çocuk / Bebek ---
    "bebek nefes almıyor", "çocuk boğuluyor",
    # --- Diğer ---
    "morarıyor", "morardı", "havale geçiriyor",
    "nöbet geçiriyor", "yılan ısırdı", "akrep soktu",
]

# ─── Global Değişkenler (lazy-load ile doldurulur) ─────────────────────────────
_model: SentenceTransformer = None
_index: faiss.IndexFlatIP = None
_chunks: list[dict] = []


def _load_model() -> SentenceTransformer:
    """Modeli bir kez yükler, sonraki çağrılarda önbellekten döner."""
    global _model
    if _model is None:
        print("[VectorDB] Embedding modeli yükleniyor...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[VectorDB] Model hazır.")
    return _model


def _compute_data_hash() -> str:
    """
    data/ klasöründeki tüm .txt dosyalarının içerik hash'ini hesaplar.
    Veri değiştiyse cache'i geçersiz kılmak için kullanılır.
    """
    hasher = hashlib.md5()
    txt_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
    for filename in txt_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, "rb") as f:
                hasher.update(f.read())
        except Exception:
            pass
    return hasher.hexdigest()


def _save_cache(index: faiss.IndexFlatIP, chunks: list[dict], data_hash: str):
    """FAISS index ve chunk metadata'sını diske kaydeder."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        faiss.write_index(index, os.path.join(CACHE_DIR, "index.faiss"))
        with open(os.path.join(CACHE_DIR, "chunks.pkl"), "wb") as f:
            pickle.dump({"chunks": chunks, "hash": data_hash}, f)
        print(f"[VectorDB] Cache kaydedildi: {len(chunks)} chunk")
    except Exception as e:
        print(f"[VectorDB] Cache kaydetme hatası: {e}")


def _load_cache(current_hash: str):
    """
    Disk cache'ten FAISS index ve chunk'ları yükler.
    Eğer veri değişmişse (hash uyuşmuyorsa) None döner.
    """
    idx_path = os.path.join(CACHE_DIR, "index.faiss")
    chunks_path = os.path.join(CACHE_DIR, "chunks.pkl")

    if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
        return None, None

    try:
        with open(chunks_path, "rb") as f:
            cache_data = pickle.load(f)

        if cache_data.get("hash") != current_hash:
            print("[VectorDB] Veri değişmiş, cache geçersiz — yeniden oluşturulacak.")
            return None, None

        index = faiss.read_index(idx_path)
        chunks = cache_data["chunks"]
        print(f"[VectorDB] Cache'ten yüklendi: {len(chunks)} chunk")
        return index, chunks
    except Exception as e:
        print(f"[VectorDB] Cache yükleme hatası: {e}")
        return None, None


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
    Vektör veritabanını başlatır.
    Önce disk cache kontrol eder, yoksa yeniden oluşturur ve cache'ler.
    """
    global _chunks, _index

    # Veri hash'ini hesapla
    data_hash = _compute_data_hash()

    # Disk cache dene
    cached_index, cached_chunks = _load_cache(data_hash)
    if cached_index is not None and cached_chunks is not None:
        _index = cached_index
        _chunks = cached_chunks
        # Model'i de yükle (cache'ten gelse bile arama sırasında lazım)
        _load_model()
        return

    # Cache yoksa yeniden oluştur (Smart Chunker kullanarak)
    print("[VectorDB] Cache bulunamadı, yeniden oluşturuluyor...")
    _chunks = chunk_directory(DATA_DIR)

    if _chunks:
        _index = _build_index(_chunks)
        # Diske kaydet
        _save_cache(_index, _chunks, data_hash)
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

    if "yandi" in ql or "yakti" in ql or "yanik" in ql or "kaynar" in ql:
        ek.append("yanık yanma termal ısı hasarı derece")
    if "kesti" in ql or "kesiti" in ql or "kesik" in ql or "kaniyo" in ql or "kanama" in ql or "kanar" in ql:
        ek.append("kesik kanama açık yara baskı turnike")
    if "pasli" in ql or "demir" in ql or "civi" in ql or "tetanoz" in ql:
        ek.append("tetanoz enfeksiyon paslı açık yara aşı")
    if "bocek" in ql or "ari " in ql or "soktu" in ql or "isirdi" in ql or "siyan" in ql or "yilan" in ql or "akrep" in ql:
        ek.append("ısırık sokma alerji anafilaksi hayvan zehir arı")
    if "nefes" in ql or "bogul" in ql or "tikandi" in ql or "yutam" in ql:
        ek.append("boğulma heimlich astım hava yolu tıkanıklık")
    if "basim don" in ql or "bayil" in ql or "gozum karar" in ql or "bilinc" in ql:
        ek.append("bayılma senkop vertigo bilinç kaybı şok pozisyonu")
    if "carp" in ql or "dustu" in ql or "kirildi" in ql or "kaza" in ql:
        ek.append("travma kırık incinme atel kafa kanaması omurga")
    if "zehir" in ql or "ilac icti" in ql or "mide" in ql or "kustu" in ql:
        ek.append("zehirlenme kusma bulantı 114 karbon monoksit")
    if "seker" in ql or "diyabet" in ql or "hipogli" in ql:
        ek.append("şeker düşmesi hipoglisemi diyabetik koma glukoz")
    if "tansiyon" in ql or "bas agrisi" in ql:
        ek.append("hipertansiyon baş ağrısı kan basıncı felç")
    if "gogus" in ql or "kalp" in ql or "kalb" in ql:
        ek.append("kalp krizi miyokart göğüs ağrısı CPR aspirin")
    if "felc" in ql or "inme" in ql or "yuzum" in ql or "kolum" in ql:
        ek.append("felç inme FAST BE-FAST konuşma bozukluğu")
    if "cpr" in ql or "kalp masaji" in ql or "yasam destegi" in ql:
        ek.append("CPR KPR kompresyon ventilasyon temel yaşam desteği AED")
    if "bebek" in ql or "cocuk" in ql or "yeni dogan" in ql:
        ek.append("bebek çocuk pediatrik yenidoğan")
    if "burun" in ql and ("kan" in ql or "akiyor" in ql):
        ek.append("burun kanaması epistaksis sıkma dik oturma")
    if "nobeti" in ql or "sara" in ql or "epilepsi" in ql or "kasil" in ql:
        ek.append("epilepsi nöbet sara koma pozisyonu")
    if "astim" in ql or "hisilt" in ql or "inhaler" in ql:
        ek.append("astım bronkospazm inhaler nefes darlığı")
    if "kirik" in ql or "burkulma" in ql or "cikik" in ql:
        ek.append("kırık burkulma çıkık atel RICE tespit")
    if "sepsis" in ql or "enfeksiyon" in ql or "ates" in ql:
        ek.append("sepsis enfeksiyon SOFA qSOFA antibiyotik şok")
    if "travma" in ql or "kaza" in ql or "yaralanma" in ql:
        ek.append("travma ATLS ABCDE birincil değerlendirme kanama")
    if "gebe" in ql or "hamile" in ql or "dogum" in ql or "dogu" in ql:
        ek.append("gebelik doğum preeklampsi eklampsi postpartum")
    if "pnomot" in ql or "gogus" in ql or "akciger" in ql:
        ek.append("pnömotoraks hemotoraks göğüs tüpü solunum")
    if "emboli" in ql or "pulmoner" in ql or "dvt" in ql:
        ek.append("pulmoner emboli DVT tromboz nefes darlığı")
    if "toksik" in ql or "uyustur" in ql or "overdoz" in ql or "opiyat" in ql:
        ek.append("toksidrom antidot nalokson zehirlenme opiyat overdoz")
    if "tiroid" in ql or "tiroit" in ql or "guatr" in ql:
        ek.append("tiroid tirotoksikoz hipotiroidi hipertiroidi TSH")
    if "dka" in ql or "ketoasidoz" in ql:
        ek.append("diyabetik ketoasidoz DKA insülin asidoz kussmaul")
    if "kbrn" in ql or "kimyasal" in ql or "radyasyon" in ql or "biyolojik" in ql:
        ek.append("KBRN kimyasal biyolojik radyasyon nükleer dekontaminasyon")
    if "pediatr" in ql or "yenidogan" in ql or "nrp" in ql:
        ek.append("pediatrik yenidoğan PALS NRP çocuk resüsitasyon")
    if "intihar" in ql or "ozkiyim" in ql or "kendine zarar" in ql:
        ek.append("intihar özkıyım psikiyatrik acil 182 risk değerlendirme")

    if ek:
        return query + " " + " ".join(ek)
    return query


def get_context(query: str) -> tuple[str, list[dict], bool, float]:
    """
    Kullanıcı sorgusuna en alakalı metin parçalarını döndürür.

    Returns:
        context_text (str): LLM'e gönderilecek birleşik bağlam metni
        sources (list[dict]): Kaynak bilgileri (kaynak gösterimi için)
        is_critical (bool): Sorgu kritik/acil durumu tetikliyor mu?
        avg_confidence (float): Ortalama benzerlik skoru (0.0-1.0)
    """
    global _chunks, _index

    # Sistem hazır değilse başlat
    if not _chunks or _index is None:
        initialize()

    if not _chunks or _index is None:
        return "", [], is_critical_query(query), 0.0

    is_critical = is_critical_query(query)

    # Sorguyu zenginleştir (Keyword/Synonym bazlı)
    enhanced_query = _enhance_query(query)

    # Sorguyu embed et
    model = _load_model()
    query_embedding = model.encode([enhanced_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # FAISS arama
    actual_k = min(TOP_K, len(_chunks))
    assert _index is not None
    scores, indices = _index.search(query_embedding, actual_k)

    results = []
    # pyre-ignore
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue

        # Benzerlik eşiği kontrolü
        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = _chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        results.append(chunk)

    if not results:
        return "", [], is_critical, 0.0

    # Confidence score hesapla
    avg_confidence = sum(r["similarity_score"] for r in results) / len(results)

    # Bağlam metnini oluştur
    context_parts = []
    sources = []
    for i, r in enumerate(results, 1):
        section_info = r.get("section", "")
        sub_info = r.get("subsection", "")
        label = f"{r['source']}"
        if section_info:
            label += f" > {section_info}"
        if sub_info:
            label += f" > {sub_info}"

        context_parts.append(f"[Kaynak {i}: {label}]\n{r['text']}")
        sources.append({
            "title": r.get("title", ""),
            "source": r["source"],
            "section": section_info,
            "subsection": sub_info,
            "score": r["similarity_score"],
            "keywords": r.get("keywords", []),
        })

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources, is_critical, avg_confidence


# ─── Modül Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize()
    test_queries = [
        "Arı soktu ne yapmalıyım?",
        "Kalp durmuş ne yapacağım?",
        "Bebekte CPR nasıl yapılır?",
        "Bitcoin nedir?",
        "Yanık oldu ne yapmalıyım?",
    ]
    for q in test_queries:
        ctx, srcs, crit, conf = get_context(q)
        print(f"\n{'='*60}")
        print(f"Sorgu: {q}")
        print(f"Kritik: {crit}")
        print(f"Confidence: {conf:.2f}")
        print(f"Kaynak sayısı: {len(srcs)}")
        if srcs:
            print(f"En iyi kaynak: {srcs[0]['title']} (score: {srcs[0]['score']:.3f})")
        ctx_str = str(ctx)
        print(f"Bağlam: {ctx_str[:200]}...")
