# -*- coding: utf-8 -*-
"""
backend/rag_engine.py - HayatKurtaran AI RAG Motoru
google-genai SDK + Otomatik retry + Model fallback
"""
import os
import sys
import io
import time
import logging

# Windows encoding fix
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # pyre-ignore
from backend.vector_db import get_context, initialize  # pyre-ignore
from backend.config import (  # pyre-ignore
    GEMINI_MODEL_CANDIDATES,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_WAIT_SECONDS,
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sen 'HayatKurtaran AI' adinda, Turkce konusan acil ilk yardim ve saglik asistanisin.\n"
    "KURALLAR:\n"
    "1. SADECE sana verilen Bilgi Baglami icerisindeki bilgilere dayan.\n"
    "2. Baglamda OLMAYAN bir konuyu, farkli veya bilinmeyen bir acil hastalik belirtisini sorarlarsa 'Bilgim disinda, acil teshis veya yonlendirme yapamam. Lutfen bu belirtiler icin DERHAL 112 ambulansi arayin.' de.\n"
    "3. Cevaplarin KISA, NET ve maddeli (1. 2. 3. seklinde) olsun. Maksimum 5-6 adimi gecmesin.\n"
    "4. Kesinlikle tibbi teshis veya recete yazilimi yapma, sadece ilk yardim adimlarini aktar.\n"
    "5. Her cevabinin sonuna sadece u yariyi ekle: 'Bu bilgi yapay zeka tarafindan uretilmistir. Acil durumda daima 112\\'yi arayin.'\n"
)

CRITICAL_WARNING = (
    "### KRITIK ACIL DURUM ALGILANDI!\n\n"
    "**HEMEN 112'YI ARAYIN!**\n\n"
    "Asagidaki bilgiler ambulans beklerken uygulanabilir.\n\n---\n\n"
)

NO_CONTEXT_RESPONSE = (
    "Bilgim dışında, kesin teşhis veya tıbbi yönlendirme yapamam.\n\n"
    "**Lütfen farklı veya bilinmeyen acil hastalık belirtileri için DERHAL 112 ambulansı arayın!**\n\n"
    "Bu bilgi yapay zeka tarafından üretilmiştir. Acil durumda daima 112'yi arayın."
)

QUOTA_ERROR_RESPONSE = (
    "Sistemimizde anlık yoğunluk yaşanmaktadır (API Kotası Doldu). "
    "Lütfen beklemeyin, **DERHAL 112'yi arayarak ambulans çağırın!**\n\n"
    "Bu bilgi yapay zeka tarafından üretilmiştir."
)

_client = None
_initialized = False


def _setup_gemini():
    global _client
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY bulunamadi! .env dosyaniza GEMINI_API_KEY=... ekleyin."
        )
    from google import genai  # pyre-ignore
    _client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("[RAG] Gemini client OK")


def startup():
    global _initialized
    if _initialized:
        return
    logger.info("[RAG] Baslatiliyor...")
    initialize()
    _setup_gemini()
    _initialized = True
    logger.info("[RAG] Hazir.")


def _call_gemini(prompt: str) -> tuple[str | None, str | None]:
    """Gemini API'yi retry ve model fallback ile cagir.

    Returns:
        (text, error_type)
        error_type: None | "quota" | "other"
    """
    last_error: Exception | None = None
    last_error_type: str | None = None

    for model_name in GEMINI_MODEL_CANDIDATES:
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                if _client is None:
                    continue
                # pyre-ignore
                response = _client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = response.text if response.text else ""
                if text.strip():
                    return text.strip(), None
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    last_error_type = "quota"
                    logger.warning(
                        "[RAG] %s kota hatasi, hemen bir sonrakine geciliyor: %s",
                        model_name,
                        err_str,
                    )
                    break  # Kota doluysa bu modeli zorlama, diğerine geç
                else:
                    last_error_type = "other"
                    logger.error("[RAG] %s hata: %s", model_name, err_str)
                    break

        logger.warning("[RAG] %s basarisiz, sonraki model deneniyor...", model_name)

    logger.error("[RAG] Tum modeller basarisiz: %s", last_error)
    return None, last_error_type


def generate_answer(query):
    global _initialized
    if not _initialized:
        startup()

    context_text, sources, is_critical = get_context(query)

    if not context_text:
        return {
            "answer": (CRITICAL_WARNING if is_critical else "") + NO_CONTEXT_RESPONSE,
            "sources": [],
            "is_critical": is_critical,
            "has_context": False,
        }

    full_prompt = (
        SYSTEM_PROMPT + "\n\n---\nBilgi Baglami:\n"
        + context_text
        + "\n---\n\nKullanicinin Sorusu: " + query
        + "\n\nYukaridaki baglama dayanarak kisa ve maddeli cevap ver."
    )

    raw, error_type = _call_gemini(full_prompt)
    if not raw:
        # a) Kota / yoğunluk hatası: kullaniciya net quota uyarisi
        if error_type == "quota":
            raw = QUOTA_ERROR_RESPONSE
        else:
            # b) Diger hatalarda: FAISS baglamindan ham kayit gosteren fallback
            first_chunk_only = (
                context_text.split("\n\n---\n\n")[0] if context_text else ""
            )
            raw = (
                "⚠️ **SİSTEM UYARISI: YAPAY ZEKA SUNUCULARI ŞU AN ERİŞİLEMIYOR.**\n\n"
                "Zaman kaybetmemeniz için bilgi tabanımızdan eşleşen "
                "**EN ÖNEMLİ DOĞRUDAN DÖKÜMAN KAYDI** aşağıda listelenmiştir:\n\n"
                f"> {first_chunk_only}\n\n"
                "*(Lütfen dikkat: Bu metinler özetlenmemiş, orijinal kaynaktır. "
                "ACİL BİR DURUMSA DERHAL 112'Yİ ARAYIN!)*"
            )

    final = (CRITICAL_WARNING if is_critical else "") + raw

    return {
        "answer": final,
        "sources": sources,
        "is_critical": is_critical,
        "has_context": True,
    }


if __name__ == "__main__":
    startup()
    r = generate_answer("Ari soktu ne yapmaliyim?")
    answer = str(r.get("answer", ""))
    print("\nCEVAP:\n", answer)
