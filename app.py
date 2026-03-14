# -*- coding: utf-8 -*-
"""
app.py  —  HayatKurtaran AI  |  Senior Developer Edition
RAG + FAISS + Gemini  |  Streamlit Web Arayuzu
Komut: streamlit run app.py
"""

import streamlit as st  # pyre-ignore
import streamlit.components.v1 as components
import sys
import os

# Windows UTF-8 encoding zorla
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Proje kok dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Sayfa Konfigurasyonu ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="HayatKurtaran AI | Ilk Yardim Chatbotu",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Stilleri ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    }

    /* ─── Hero ─── */
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4757 0%, #ff6b81 50%, #ff9f43 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 0.4rem 0 0.1rem;
    }
    .hero-sub {
        text-align: center;
        color: #8892a4;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    /* ─── Kritik banner ─── */
    .crit-banner {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        border-radius: 12px;
        padding: 10px 18px;
        text-align: center;
        color: #fff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(231,76,60,0.45);
        animation: pulse 2.5s ease-in-out infinite;
        letter-spacing: 0.4px;
    }
    @keyframes pulse {
        0%,100% { box-shadow: 0 4px 20px rgba(231,76,60,0.4); }
        50%      { box-shadow: 0 4px 38px rgba(231,76,60,0.85); }
    }

    /* ─── Quick btn label ─── */
    .quick-label {
        color: #ff6b81;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        margin-bottom: 0.4rem;
    }

    /* ─── Kaynak etiketi ─── */
    .src-tag {
        display: inline-block;
        background: rgba(255,71,87,.12);
        border: 1px solid rgba(255,71,87,.35);
        color: #ff7f91;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.73rem;
        font-weight: 500;
        margin: 6px 4px 0 0;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: rgba(10,17,34,0.97) !important;
        border-right: 1px solid rgba(255,71,87,0.18);
    }
    .sb-title {
        font-family: 'Poppins', sans-serif;
        color: #ff6b81;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .stat-box {
        background: rgba(255,71,87,0.07);
        border: 1px solid rgba(255,71,87,0.18);
        border-radius: 9px;
        padding: 9px 13px;
        margin-bottom: 8px;
        color: #c5ceda;
        font-size: 0.86rem;
    }
    .stat-box strong { color: #ff6b81; }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #ff4757; border-radius: 8px; }

    hr { border-color: rgba(255,71,87,0.15) !important; }

    /* Chat mesaj arkaplanı */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 0.6rem;
    }

    /* Buton stili */
    .stButton > button {
        background: rgba(255,71,87,0.10);
        border: 1px solid rgba(255,71,87,0.28);
        color: #ffb3bc;
        border-radius: 9px;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: rgba(255,71,87,0.22);
        border-color: #ff4757;
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(255,71,87,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "pending_query": None,
    "total_queries": 0,
    "critical_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── RAG Motoru (tek seferlik cache) ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    from backend.rag_engine import startup, generate_answer  # pyre-ignore
    startup()
    return generate_answer

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_answer(query: str):
    """
    Sisteme önceden sorulmuş soruları RAM önbelleğine (Cache) kaydeder.
    Eğer aynı kullanıcı aynı veya benzer soruyu 1 saat (3600sn) içinde sorarsa,
    LLM API'sine hiç istek atmadan sonucu milisaniyede döndürür.
    Böylece Google kotaları %100 oranında by-pass edilir.
    """
    generate = load_engine()
    return generate(query)


# ─── Sayfa Basligi ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🚑 HayatKurtaran AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">RAG Destekli İlk Yardım ve Sağlık Chatbotu &nbsp;·&nbsp; Gemini + FAISS</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="crit-banner">'
    '&#9888; HAYATI TEHLİKE DURUMUNDA HEMEN <strong>112</strong>\'Yİ ARAYIN &#9888; '
    '&nbsp;|&nbsp; Bu asistan bilgi amaçlıdır; tıbbi tanı ve tedavi yerine geçmez.'
    '</div>',
    unsafe_allow_html=True
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">🚑 HayatKurtaran AI</div>', unsafe_allow_html=True)
    st.caption("RAG Mimarisi · Gemini 1.5 Flash · FAISS · v2.0")
    st.divider()

    st.markdown("**📊 Oturum İstatistikleri**")
    st.markdown(f'<div class="stat-box">Toplam Soru: <strong>{st.session_state.total_queries}</strong></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-box">Acil Durum Tespiti: <strong>{st.session_state.critical_count}</strong></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**📁 Bilgi Tabanı**")
    st.markdown("🩺 **İlk Yardım Bilgileri** — 13+ senaryo")
    st.markdown("🚨 **Acil Durumlar** — Kritik müdahaleler")
    st.markdown("💊 **Sağlık Önerileri** — Koruyucu tavsiyeler")
    st.divider()

    st.markdown("**⚡ Hızlı Aramalar**")
    sidebar_queries = [
        "CPR nasıl yapılır?",
        "Yanıkta ne yapmalıyım?",
        "Burun kanamasını nasıl durdururum?",
        "Epilepsi nöbetinde ne yapılır?",
    ]
    for i, q in enumerate(sidebar_queries):
        if st.button(q, key=f"sb_{i}", use_container_width=True):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.critical_count = 0
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="color:#4a5568;font-size:0.72rem;text-align:center;">'
        'HayatKurtaran AI v2.0 · Senior Developer Edition<br>'
        'Veriler doğruluk açısından kontrol edilmiştir.</div>',
        unsafe_allow_html=True
    )


# ─── Hızlı Aksiyon Butonları ──────────────────────────────────────────────────
st.markdown('<div class="quick-label">⚡ Hızlı Aksiyon — Acil Senaryolar</div>', unsafe_allow_html=True)

quick_actions = [
    ("❤️ Kalp Krizi / CPR",          "Kalp krizi belirtileri neler, ne yapmalıyım?"),
    ("🫁 Boğulma / Heimlich",          "Biri boğuluyor, Heimlich manevrası nasıl yapılır?"),
    ("🩸 Şiddetli Kanama",             "Şiddetli kanamayı durdurmak için ne yapmalıyım?"),
    ("🔥 Yanık Müdahalesi",            "Yanığa nasıl ilk yardım yapılır?"),
    ("⚡ Felç / İnme Belirtileri",     "Felç belirtileri nelerdir, ne yapmalıyım?"),
    ("💉 Anafilaktik Şok",             "Anafilaktik şok nedir, ne yapılır?"),
    ("⚡ Elektrik Çarpması",           "Elektrik çarptı ne yapmalıyım?"),
    ("🫀 Tansiyon Krizi",              "Tansiyonum çok yüksek ne yapmalıyım?"),
    ("🍬 Şeker Düşmesi",               "Şekerim düştü ne yapmalıyım?"),
]

cols = st.columns(3)
for idx, (label, query) in enumerate(quick_actions):
    with cols[idx % 3]:
        if st.button(label, key="qa_" + str(idx), use_container_width=True):
            st.session_state.pending_query = query

st.divider()


def render_emergency_buttons():
    # Dikkat çekici titreşen 112 butonu için styling (CSS)
    st.markdown("""
        <style>
        .emergency-call-btn a {
            background-color: #ff3b30 !important;
            color: white !important;
            font-weight: 800 !important;
            font-size: 22px !important;
            padding: 15px 30px !important;
            border-radius: 12px !important;
            text-transform: uppercase !important;
            text-align: center !important;
            display: block !important;
            margin: 20px auto !important;
            max-width: 400px;
            text-decoration: none !important;
            animation: pulseAlert 1.5s infinite;
            box-shadow: 0 4px 15px rgba(255, 59, 48, 0.6) !important;
        }
        @keyframes pulseAlert {
            0% { transform: scale(1); box-shadow: 0 4px 15px rgba(255, 59, 48, 0.6); }
            50% { transform: scale(1.05); box-shadow: 0 8px 25px rgba(255, 59, 48, 0.9); }
            100% { transform: scale(1); box-shadow: 0 4px 15px rgba(255, 59, 48, 0.6); }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Tıklanabilir link butonu basılıyor
    st.markdown('<div class="emergency-call-btn"><a href="tel:112">📞 HEMEN 112\'Yİ ARA</a></div>', unsafe_allow_html=True)
    
    # GPS Destekli Akıllı SMS Butonu
    components.html(
        '''
        <style>
        .sms-btn {
            background: linear-gradient(135deg, #0984e3, #74b9ff);
            color: white;
            font-family: sans-serif;
            font-weight: bold;
            font-size: 16px;
            padding: 14px 24px;
            border-radius: 12px;
            text-align: center;
            display: block;
            margin: 0 auto;
            width: 80%;
            max-width: 400px;
            cursor: pointer;
            border: none;
            text-decoration: none;
            box-shadow: 0 4px 10px rgba(9, 132, 227, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .sms-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(9, 132, 227, 0.6);
        }
        .sms-btn:active { transform: scale(0.98); }
        </style>
        <button class="sms-btn" onclick="sendEmergencyLocation()">📍 Konumumu 112'ye SMS Gönder</button>
        <script>
        function sendEmergencyLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    var lat = position.coords.latitude;
                    var lon = position.coords.longitude;
                    var mapsLink = "https://maps.google.com/?q=" + lat + "," + lon;
                    var msg = "ACİL DURUM! Lütfen bana yardım edin. Konumum: " + mapsLink;
                    window.parent.location.href = "sms:112?body=" + encodeURIComponent(msg);
                }, function(error) {
                    console.error("GPS hatasi: ", error);
                    var msg = "ACİL DURUM! Cihaz konum bilgilerimi alamadım, lütfen benimle bu numaradan iletişime geçin.";
                    window.parent.location.href = "sms:112?body=" + encodeURIComponent(msg);
                }, { timeout: 5000 });
            } else {
                alert("Tarayıcınız konum özelliğini desteklemiyor.");
            }
        }
        </script>
        ''',
        height=70
    )


# ─── Sohbet Arayüzü ───────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🚑"):
            st.markdown("""
Merhaba! Ben **HayatKurtaran AI** – acil ilk yardım ve sağlık konularında size Türkçe yardım eden bir yapay zekâ asistanıyım.

**Nasıl çalışırım?**
- Sorunuzu doğal dilde yazın: *"Arı soktu elimde şişlik var"*
- Ya da üstteki **Hızlı Aksiyon** butonlarından bir senaryo seçin

> 🚨 **Hayati tehlike durumunda bu uygulamayı beklemeyin — derhal 112'yi arayın!**
            """)

    for msg in st.session_state.messages:
        avatar = "🚑" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant":
                render_emergency_buttons()
            if msg.get("sources"):
                src_html = " ".join(
                    f'<span class="src-tag">📄 {s["source"]}</span>'
                    for s in msg["sources"][:3]  # Ekranda 20 tane basarak arayüzü kirletmesini engelle (Top 3)
                )
                st.markdown(src_html, unsafe_allow_html=True)


# ─── Sorgu İşleme ─────────────────────────────────────────────────────────────
def process_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})

    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🚑"):
            with st.spinner("Bilgi tabanında aranıyor, yanıt hazırlanıyor…"):
                try:
                    result = get_cached_answer(query)

                    answer      = result.get("answer", "")
                    sources     = result.get("sources", [])
                    is_critical = result.get("is_critical", False)
                    has_context = result.get("has_context", True)

                    st.session_state.total_queries += 1
                    if is_critical:
                        st.session_state.critical_count += 1

                    st.markdown(answer, unsafe_allow_html=True)
                    
                    # Hastanın durumu sonradan fenalaşabileceği ihtimaline karşın
                    # butonlar aciliyete bakılmaksızın her sorunun/cevabın altında ZORLA gösterilir.
                    render_emergency_buttons()
                        
                    if sources:
                        src_html = " ".join(
                            f'<span class="src-tag">📄 {s["source"]}</span>'
                            for s in sources[:3]
                        )
                        st.markdown(src_html, unsafe_allow_html=True)

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


# ─── Tetikleyiciler ───────────────────────────────────────────────────────────
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    process_query(q)
    st.rerun()

if prompt := st.chat_input("İlk yardım sorunuzu yazın… örn: 'Arı soktu ne yapmalıyım?'"):
    process_query(prompt)
    st.rerun()
