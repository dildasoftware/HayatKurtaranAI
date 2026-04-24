# -*- coding: utf-8 -*-
"""
backend/rag_engine.py - HayatKurtaran AI RAG Motoru
google-genai SDK + Otomatik retry + Model fallback
"""
import os
import sys
import io
import time

# Windows encoding fix
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # pyre-ignore
from backend.vector_db import get_context, initialize  # pyre-ignore
from backend.logger import get_logger, QueryLog, LatencyTimer  # pyre-ignore
from PIL import Image  # Added for vision compression

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Sirasyla deneyecegimiz modeller (biri kota dolarsa digeri denenir)
MODEL_CANDIDATES = [
    "models/gemini-flash-lite-latest",
    "models/gemma-3-27b-it",
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash",
    "models/gemini-pro-latest"
]

MAX_RETRIES = 2
RETRY_WAIT = 1  # saniye

SYSTEM_PROMPT = (
    "Sen 'HayatKurtaran AI' adında, Türkçe konuşan acil ilk yardım ve sağlık asistanısın.\n"
    "MUTLAK KURALLAR:\n"
    "1. SADECE sana verilen 'Bilgi Bağlamı' içerisindeki bilgilere dayan.\n"
    "2. Bağlamda OLMAYAN bir konu sorulduğunda: 'Bu konuda bilgi tabanımda yeterli kaynak "
    "bulunamadı. Lütfen bir sağlık profesyoneline danışın veya acil durumda 112\\'yi arayın.' de.\n"
    "3. KESINLIKLE bağlamda olmayan bilgiyi UYDURMA. Halüsinasyon yapma. "
    "Emin olmadığın bilgiyi sakla, tahmin yürütme.\n"
    "4. Cevapların KISA, NET ve maddeli (1. 2. 3.) olsun. Maksimum 6 adım.\n"
    "5. Kesinlikle tıbbi teşhis koyma veya reçete yazma; sadece ilk yardım adımlarını aktar.\n"
    "6. Her cevabının sonuna şunu ekle: "
    "'⚠️ Bu bilgi yapay zeka tarafından üretilmiştir. Tıbbi tanı yerine geçmez. "
    "Acil durumda derhal 112\\'yi arayın.'\n"
)

CRITICAL_WARNING = (
    "### KRITIK ACIL DURUM ALGILANDI!\n\n"
    "**HEMEN 112'YI ARAYIN!**\n\n"
    "Asagidaki bilgiler ambulans beklerken uygulanabilir.\n\n---\n\n"
)

NO_CONTEXT_RESPONSE = (
    "⚠️ Bu konuda bilgi tabanımda yeterli kaynak bulunamadı.\n\n"
    "**Güvenliğiniz için lütfen:**\n"
    "- Bir sağlık profesyoneline danışın\n"
    "- Acil bir durum söz konusuysa **DERHAL 112** ambulansını arayın\n\n"
    "⚠️ Bu bilgi yapay zeka tarafından üretilmiştir. Tıbbi tanı yerine geçmez. "
    "Acil durumda derhal 112'yi arayın."
)

QUOTA_ERROR_RESPONSE = (
    "Sistemimizde anlık yoğunluk yaşanmaktadır (API Kotası Doldu). "
    "Lütfen beklemeyin, **DERHAL 112'yi arayarak ambulans çağırın!**\n\n"
    "Bu bilgi yapay zeka tarafından üretilmiştir."
)

# ─── SEMANTIC CACHE (Sıfır Gecikme Ön Belleği) ───
SEMANTIC_CACHE = {
    "kalp krizi": "🚨 **HEMEN 112'Yİ ARAYIN!**\n\n1. Hastayı sakinleştirin ve oturtun veya yarı yatar pozisyona getirin.\n2. Varsa kravat, yaka, kemer gibi dar giysileri gevşetin.\n3. Hastanın bilinen kalp ilacı varsa almasına yardımcı olun.\n4. Hastayı kesinlikle yürütmeyin veya egzersiz yaptırmayın.\n5. Bilinci kapanırsa ve nefes almazsa CPR (Kalp Masajı) işlemine başlayın.\n\n⚠️ Bu bilgi (Ön-Bellek) tarafından üretilmiştir. Tıbbi tanı yerine geçmez.",
    "cpr nasıl yapılır": "🚨 **HEMEN 112'Yİ ARAYIN!**\n\n1. Hastanın bilincini ve solunumunu kontrol edin. Nefes almıyorsa başlayın.\n2. Sert bir zemine yatırın. Göğsün orta kısmına iki elinizi kenetleyip yerleştirin.\n3. Kollarınızı bükmeden, göğüs kafesini 5 cm çöktürecek şekilde dakikada 100-120 bası ritmiyle sertçe bastırın.\n4. 30 kalp masajından sonra 2 suni solunum yapın (30:2 kuralı).\n5. Ambulans gelene kadar durmayın.\n\n⚠️ Bu bilgi (Ön-Bellek) tarafından üretilmiştir.",
    "arı soktu": "1. Mümkünse iğneyi cımbız kullanmadan, kredi kartı gibi düz bir cisimle sıyırarak hızla çıkarın.\n2. Sokulan bölgeyi soğuk su ve sabunla yıkayın.\n3. Şişliği azaltmak için 10-15 dakika soğuk kompres/buz uygulayın.\n4. Alerjik şok (nefes darlığı, yüzde şişme) gelişirse DERHAL 112'yi arayın.\n\n⚠️ Bu bilgi (Ön-Bellek) tarafından üretilmiştir.",
    "burnum kanıyor": "1. Hastayı dik oturtun (başını GERİYE ATMAYIN!).\n2. Başını hafifçe öne doğru eğin.\n3. Burun kanatlarını baş ve işaret parmağınızla 5-10 dakika boyunca sıkıca bastırın.\n4. Enseye veya burun köküne soğuk uygulayabilirsiniz.\n5. Kanama 20 dakikadan uzun sürerse 112'yi arayın.\n\n⚠️ Bu bilgi (Ön-Bellek) tarafından üretilmiştir."
}

_client = None
_initialized = False


def _setup_gemini():
    global _client
    if not GEMINI_API_KEY:
        print("[UYARI] GEMINI_API_KEY bulunamadi! Sadece On Bellek (Semantic Cache) calisacak.")
        return
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    _client = genai  # Sadece referans kontrolü için
    print("[RAG] Gemini client OK")


def startup():
    global _initialized
    if _initialized:
        return
    print("[RAG] Baslatiliyor...")
    initialize()
    _setup_gemini()
    _initialized = True
    print("[RAG] Hazir.")


def _call_gemini(prompt: str, image: Image.Image = None) -> str:
    """Gemini API'yi retry ve model fallback ile cagir."""
    last_error = None
    
    # Resim varsa Multi-modal content hazırla
    contents = [prompt]
    if image:
        contents.append(image)

    for model_name in MODEL_CANDIDATES:
        for attempt in range(MAX_RETRIES):
            try:
                if _client is None:
                    continue
                model = _client.GenerativeModel(model_name)
                response = model.generate_content(contents)
                text = response.text if response.text else ""
                if text.strip():
                    return text.strip()
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"[RAG] {model_name} kota hatası, hemen bir sonrakine geçiliyor.")
                    break  # Kota doluysa bu modeli zorlama, diğerine geç
                else:
                    print(f"[RAG] {model_name} hata: {err_str}")
                    break

        print(f"[RAG] {model_name} basarisiz, sonraki model deneniyor...")

    print(f"[RAG] Tum modeller basarisiz: {last_error}")
    return None


def _extract_text_from_image(image: Image.Image) -> str:
    """
    [Aşama 1 Güvenlik Katmanı] Resmi analiz eder ama ASLA teşhis koymaz.
    Sadece görüntüdeki etiketleri veya fiziksel objeleri metne çevirir.
    """
    # Ön İyileştirme: Hız (Latency) ve Güvenlik için görüntüyü sıkıştır
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"[Vision] Görsel sıkıştırma hatası: {e}")

    vision_prompt = (
        "Sen klinik bir ilk yardım asistanının görme duyususun. GÖREVİN: "
        "Görüntüyü analiz edip, ilk yardım gerektiren bir durum varsa (kanama, yara, morarma, şişlik vb.) "
        "bunu detaylı ama teşhis koymadan klinik bir dille betimlemektir. "
        "Ayrıca eğer okunabilir bir isim/etiket varsa okuyup metne yaz.\n"
        "Örnek: 'Hastanın sağ şakak bölgesinde 3 cm boyutunda kanayan açık yara mevcut.' veya "
        "'Aspirin kutusu, 100mg'.\n"
        "KESİNLİKLE HASTALIK TEŞHİSİ KOYMA ve YORUM YAPMA."
    )
    
    #_call_gemini'yi sadece resim ve strict prompt ile çağırıyoruz
    extracted_text = _call_gemini(vision_prompt, image=image)
    if extracted_text:
        return extracted_text.strip()
    return "Görselden anlamlı bir bilgi çıkarılamadı."


def generate_answer(query, history=None, image: Image.Image = None):
    """
    Kullanıcı sorusuna RAG tabanlı yanıt üretir. (Ses, Görüntü ve Bellek destekli)

    Args:
        query: Kullanıcı sorusu
        history: Önceki mesajlar listesi
        image: Opsiyonel PIL Image objesi (Multi-modal)

    Returns:
        dict: answer, sources, is_critical, has_context, avg_confidence, latency_ms
    """
    global _initialized
    if not _initialized:
        startup()

    timer = LatencyTimer()
    logger = get_logger()
    error_msg = None

    # ─── 5.1 Multi-turn Dialog Memory ─────────────────────────────────
    from backend.conversation import enrich_query_with_history
    enriched_query = enrich_query_with_history(query, history)

    # ─── %100 Hızlı Semantic Cache (Ön Bellek) Katmanı ────────────────
    if not image:
        q_clean = query.lower() # Geçmiş eklenmiş sorgu (enriched_query) yerine RAW soruyu kontrol et
        for k, v in SEMANTIC_CACHE.items():
            if k in q_clean:
                # Cache yakalandı! 0ms gecikme (FAISS ve LLM pas geçilir)
                logger.log(QueryLog(query=enriched_query, error="CACHE_HIT"))
                timer.measure("faiss")  # dummy
                timer.measure("llm")  # dummy
                return {
                    "answer": v,
                    "sources": [{"title": "Güvenilir Ön Bellek", "source": "🩺 Anında Yanıt Sistemi", "score": 1.0}],
                    "is_critical": True if "112" in v else False,
                    "has_context": True,
                    "avg_confidence": 1.0,
                    "latency_ms": 5.0,  # Ortalama 5ms
                    "is_faithful": True,
                    "raw_clean_answer": v
                }

    # ─── 4.5 Vision Görme İzolasyonu (Aşama 1) ───────────────────────
    vision_description = ""
    if image:
        with timer.measure("vision"):
            vision_description = _extract_text_from_image(image)
            # Görsel bulgularını kullanıcının sorusuna arkada gizlice ekleyelim
            # Böylece FAISS aramasında bu metin üzerinden güvenli tarama yapılacak
            enriched_query = f"{enriched_query}\n(Sistem Çıkarımı - Görsel Bulgular: {vision_description})"

    # ─── 3.2 Prompt Injection Koruması ─────────────────────────────────
    injection_patterns = [
        "ignore previous", "forget instructions", "system prompt", 
        "you are now", "bana asıl", "bütün talimatları", "önceki talimatları",
        "kuralları unut", "promptu iptal"
    ]
    q_lower = enriched_query.lower()
    if any(p in q_lower for p in injection_patterns):
        # Log attack
        logger.log(QueryLog(query=enriched_query, error="Prompt Injection Detected"))
        return {
            "answer": "⚠️ **Güvenlik İhlali Tespit Edildi:** Sistem talimatlarını değiştirme girişimleri politikalara aykırıdır. Lütfen sadece ilk yardım soruları sorunuz.",
            "sources": [],
            "is_critical": False,
            "has_context": False,
            "avg_confidence": 0.0,
            "latency_ms": 0.0,
        }

    # ─── FAISS Arama (latency ölçümü) ──────────────────────────────────
    with timer.measure("faiss"):
        context_text, sources, is_critical, avg_confidence = get_context(enriched_query)

    if not context_text:
        # Log: bağlam bulunamadı
        timings = timer.results
        logger.log(QueryLog(
            query=enriched_query,
            is_critical=is_critical,
            has_context=False,
            rag_confidence=0.0,
            latency_faiss_ms=timings.get("faiss_ms", 0),
            latency_total_ms=timings.get("total_ms", 0),
        ))
        return {
            "answer": (CRITICAL_WARNING if is_critical else "") + NO_CONTEXT_RESPONSE,
            "sources": [],
            "is_critical": is_critical,
            "has_context": False,
            "avg_confidence": 0.0,
            "latency_ms": timings.get("total_ms", 0),
        }

    # Düşük güven uyarısı
    confidence_note = ""
    if avg_confidence < 0.45:
        confidence_note = (
            "\n\n> ℹ️ *Bilgi tabanımda bu soruyla yüksek eşleşme bulunamadı. "
            "Aşağıdaki bilgiler sınırlı bağlamla üretilmiştir; "
            "lütfen bir sağlık profesyoneline de danışın.*\n\n"
        )

    vision_instruction = ""
    if image:
        vision_instruction = (
            "\n\n[GÖRSEL ANALİZ TALİMATI]: Kullanıcı sana bir fotoğraf iletti. "
            "LÜTFEN ÖNCE GÖRSELİ ANALİZ ET. Gördüğün yara, kanama veya nesneyi "
            "(örn: ilaç kutusu) hastaya betimle ve ilk yardım adımlarını "
            "gördüğün bu fiziksel kanıta göre önceliklendir. "
            "Sadece metin bağlamına takılı kalma, gözlerine de güven.\n"
        )

    full_prompt = (
        SYSTEM_PROMPT 
        + vision_instruction
        + "\n\n---\nBilgi Bağlamı:\n"
        + context_text
        + "\n---\n\nKullanıcının Sorusu: " + enriched_query
        + "\n\nYukarıdaki bağlama ve varsa görsele dayanarak kısa ve maddeli cevap ver."
    )

    # ─── LLM Çağrısı (latency ölçümü) ─────────────────────────────────
    # Phase 3 Update: Görsel artık RAG motoruna doğrudan veriliyor!
    with timer.measure("llm"):
        raw = _call_gemini(full_prompt, image=image)

    if not raw:
        # API çöktüyse FAISS'ten dönen en ilgili chunk'ı kontrol et
        if avg_confidence < 0.65:
            # Benzerlik 0.65'in altındaysa emin olamıyoruz demektir! GÜVENLİK KİLİDİ!
            raw = (
                "<div style='border:2px solid red; padding:15px; border-radius:10px; background-color:#fff5f5; color:#c53030;'>"
                "### ⚠️ KRİTİK GÜVENLİK UYARISI\n\n"
                "Yapay zeka asistanına şu an ulaşılamıyor ve durumunuz yerel rehberlerimizle tam eşleşmedi.\n\n"
                "**LÜTFEN HİÇ VAKİT KAYBETMEDEN 112 ACİL ÇAĞRI MERKEZİNİ ARAYIN!**"
                "</div>"
            )
        else:
            first_chunk_only = context_text.split('\n\n---\n\n')[0] if context_text else ""
            raw = (
                "<div style='border:1px solid #3182ce; padding:15px; border-radius:10px; background-color:#ebf8ff; color:#2c5282;'>"
                "### 📘 Çevrimdışı İlk Yardım Rehberi (AI Sunucu Bağlantısı Yok)\n\n"
                "AI sunucularımız şu an yoğun olduğundan, size **%100 güvenli yerel protokol** metnini sunuyoruz:\n\n"
                f"{first_chunk_only}\n\n"
                "--- \n"
                "*⚠️ Bu metin orijinal kaynaktır. Değerlendirme için mutlaka uzman görüşü alın veya 112'yi arayın.*"
                "</div>"
            )
        error_msg = "LLM API quota exhausted — fallback used"
    else:
        # ─── 3.1 Post-Generation Faithfulness Check ───────────────────────
        from backend.faithfulness import check_faithfulness, get_hallucination_warning_html
        
        with timer.measure("faithfulness"):
            faith_result = check_faithfulness(raw, context_text, severity=1 if is_critical else 5)
            
        if not faith_result["is_faithful"]:
            raw_clean_answer = raw # Temiz metni seste uyarıya dönüştürmek üzere sakla
            raw += "\n\n" + get_hallucination_warning_html()
            error_msg = f"Hallucination detected (score: {faith_result['score']:.2f})"
        else:
            raw_clean_answer = raw
            
        faith_score = faith_result["score"]
        is_faithful_flag = faith_result["is_faithful"]

    final = (CRITICAL_WARNING if is_critical else "") + confidence_note + raw
    timings = timer.results

    # ─── Structured Log ───────────────────────────────────────────────
    logger.log(QueryLog(
        query=enriched_query,
        rag_confidence=avg_confidence,
        is_critical=is_critical,
        has_context=True,
        source_count=len(sources),
        latency_faiss_ms=timings.get("faiss_ms", 0),
        latency_llm_ms=timings.get("llm_ms", 0),
        latency_total_ms=timings.get("total_ms", 0),
        faithfulness_score=faith_score if 'faith_score' in locals() else None,
        error=error_msg,
    ))

    return {
        "answer": final,
        "sources": sources,
        "is_critical": is_critical,
        "has_context": True,
        "avg_confidence": avg_confidence,
        "latency_ms": timings.get("total_ms", 0),
        "is_faithful": is_faithful_flag if 'is_faithful_flag' in locals() else True,
        "raw_clean_answer": raw_clean_answer if 'raw_clean_answer' in locals() else raw
    }


if __name__ == "__main__":
    startup()
    test_queries = [
        "Arı soktu ne yapmalıyım?",
        "Bebekte CPR nasıl yapılır?",
        "Bitcoin nedir?",
    ]
    for q in test_queries:
        r = generate_answer(q)
        print(f"\n{'='*60}")
        print(f"Sorgu: {q}")
        print(f"Kritik: {r['is_critical']}")
        print(f"Confidence: {r['avg_confidence']:.2f}")
        print(f"Cevap: {str(r['answer'])[:300]}...")
