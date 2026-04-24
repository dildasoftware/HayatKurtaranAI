# -*- coding: utf-8 -*-
"""
app.py — HayatKurtaran AI | v3.0 Senior Developer Edition
==========================================================
RAG + FAISS + Gemini · Streamlit Web Arayüzü
Komut: streamlit run app.py
"""

import streamlit as st
import sys
import os
import time
import concurrent.futures

# Windows UTF-8 encoding zorla
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Sayfa Konfigürasyonu ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="HayatKurtaran AI | İlk Yardım Chatbotu",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS & Bileşenler ─────────────────────────────────────────────────────────
from ui.styles import get_main_css, get_panic_mode_css  # noqa: E402
from ui.components import (  # noqa: E402
    render_hero,
    render_consent,
    render_sidebar,
    render_input_tools,  # Yeni
    render_quick_actions,
    render_emergency_buttons,
    render_source_tags,
    render_confidence_bar,
    render_latency_badge,
    render_disclaimer,
    render_welcome,
)
from backend.emergency_classifier import (  # noqa: E402
    classify as triage_classify,
    get_severity_badge_html,
)
from backend.voice import text_to_speech, clean_old_voices  # Yeni

st.markdown(get_main_css(), unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "pending_query": None,
    "total_queries": 0,
    "critical_count": 0,
    "accepted_terms": False,
    "active_image": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── RAG Motoru (tek seferlik cache) ──────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_engine():
    from backend.rag_engine import startup, generate_answer
    startup()
    return generate_answer


# ─── Sayfa Render ──────────────────────────────────────────────────────────────
render_hero()

# Yasal onay kontrolü
if render_consent():
    st.stop()

# Sidebar
# Sidebar
render_sidebar()
# Görsel girişi artık birleşik çubuktan yönetiliyor (st.session_state.active_image)

# Hızlı aksiyon butonları
render_quick_actions()

# Ses dosyalarını temizle (Senior Dev housekeeping)
clean_old_voices()

# ─── Sohbet Arayüzü ───────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        render_welcome()

    for msg in st.session_state.messages:
        avatar = "🚑" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant":
                render_emergency_buttons()
            if msg.get("sources"):
                src_html = " ".join(
                    f'<span class="src-tag">📄 {s["source"]}</span>'
                    for s in msg["sources"][:3]
                )
                st.markdown(src_html, unsafe_allow_html=True)


# ─── Sorgu İşleme ─────────────────────────────────────────────────────────────
def process_query(query: str):
    """Kullanıcı sorgusunu işler, yanıt üretir ve UI'a yazar."""
    st.session_state.messages.append({"role": "user", "content": query})

    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🚑"):
            with st.spinner("Bilgi tabanında aranıyor, yanıt hazırlanıyor…"):
                try:
                    # Chat geçmişini al (sözlük formatında)
                    history_list = st.session_state.messages[:-1]
                    generate = load_engine()
                    
                    # Multi-modal RAG Çağrısı (Persistent session image kullanılıyor)
                    result = generate(query, history=history_list, image=st.session_state.active_image)

                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    is_critical = result.get("is_critical", False)
                    has_context = result.get("has_context", True)
                    confidence = result.get("avg_confidence", 0.0)
                    latency = result.get("latency_ms", 0.0)

                    # Triyaj Sınıflandırması
                    triage = triage_classify(query)
                    
                    # ─── PANİK MODU (Seviye 1-2 ise arayüzü kırmızı yap) ───
                    if triage.severity <= 2:
                        st.markdown(get_panic_mode_css(), unsafe_allow_html=True)
                        # Distraction-free (Dikkat dağıtıcı sidebar ve header'ı sil)
                        st.markdown("<style>[data-testid='stSidebar'] {display: none !important;} header {display: none !important;}</style>", unsafe_allow_html=True)
                        st.session_state.critical_count += 1

                    # Render: triage badge
                    st.markdown(get_severity_badge_html(triage), unsafe_allow_html=True)

                    # ─── SESLİ GÜVENLİK KİLİDİ (AUDIO SAFETY OVERRIDE) ───
                    is_faithful = result.get("is_faithful", True)
                    raw_clean_answer = result.get("raw_clean_answer", answer)
                    
                    voice_text = ""
                    if not is_faithful:
                        # Halüsinasyon saptandıysa ekran okumak yerine Hukuki İkaz yap
                        voice_text = "Sistem bu durumu kesin olarak doğrulayamadı. Lütfen ekranı okuyun veya DERHAL 112'yi arayın."
                    else:
                        if is_critical:
                            voice_text = "Lütfen sakin olun. Bu tıbbi bir acil durumdur, hemen 112'yi arayın. İlk yardım adımları şunlardır: "
                        
                        voice_text += raw_clean_answer

                    # ─── ASENKRON TTS & SAFE STREAMING ───
                    # Sesi arka planda üretirken, metni güvenli şekilde (zaten doğrulanmış) akıt!
                    def _stream_text(text):
                        for word in text.split(" "):
                            yield word + " "
                            time.sleep(0.015)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future_voice = executor.submit(text_to_speech, voice_text)
                        # Görsel algılanabilir hızı (%90 düşürdük)
                        st.write_stream(_stream_text(answer))
                        voice_path = future_voice.result()

                    if voice_path:
                        st.audio(voice_path, format="audio/mp3", autoplay=True)

                    render_confidence_bar(confidence, has_context)
                    render_latency_badge(latency)
                    render_disclaimer()
                    render_emergency_buttons()
                    render_source_tags(sources)

                    if not has_context:
                        st.info("ℹ️ Bu soru için bilgi tabanımda doğrudan kayıt bulunamadı.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "is_critical": is_critical,
                    })

                except EnvironmentError as e:
                    msg = f"⚙️ **Yapılandırma Hatası:** {e}"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})

                except Exception as e:
                    msg = f"❌ **Sistem Hatası:** {e}\n\nLütfen `.env` dosyanızı ve internet bağlantınızı kontrol edin."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})


# ─── Giriş Araçları (Sesli & Görsel) ───
# Not: active_image değerini burada sadece render_input_tools fonksiyonunda session_state'e yazıyor.
current_img, voice_command = render_input_tools()

# ─── Tetikleyiciler ───────────────────────────────────────────────────────────
if voice_command:
    process_query(voice_command)
    st.rerun()

if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    process_query(q)
    st.rerun()

if prompt := st.chat_input("İlkyardım için bir soru yazın veya fotoğraf ekleyin..."):
    process_query(prompt)
    # Fotoğraf analiz edildikten sonra temizlensin mi? 
    # Genelde kullanıcı yeni soru sormadan önce temizlemek ister.
    st.session_state.active_image = None 
    st.rerun()
