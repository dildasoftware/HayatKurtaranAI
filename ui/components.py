# -*- coding: utf-8 -*-
"""
ui/components.py — HayatKurtaran AI UI Bileşenleri
=====================================================
Streamlit arayüzü için yeniden kullanılabilir render fonksiyonları.
Her fonksiyon tek bir UI bileşenini render eder — tek sorumluluk ilkesi.
"""

import streamlit as st
import streamlit.components.v1 as components
from ui.styles import get_emergency_button_css


# ─── Hero Section ──────────────────────────────────────────────────────────────

def render_hero():
    """Sayfa başlığı ve acil uyarı banner'ı."""
    st.markdown(
        '<div class="hero-title">🚑 HayatKurtaran AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">'
        'RAG Destekli İlk Yardım ve Sağlık Chatbotu &nbsp;·&nbsp; Gemini + FAISS'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="crit-banner">'
        '&#9888; HAYATI TEHLİKE DURUMUNDA HEMEN <strong>112</strong>\'Yİ ARAYIN &#9888; '
        '&nbsp;|&nbsp; Bu asistan bilgi amaçlıdır; tıbbi tanı ve tedavi yerine geçmez.'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Terms & Consent ──────────────────────────────────────────────────────────

def render_consent():
    """Kullanım koşulları onay ekranı. Onay verilmediyse True döner (stop sinyali)."""
    if st.session_state.get("accepted_terms", False):
        return False  # Zaten onaylandı, devam et

    placeholder = st.empty()
    with placeholder.container():
        st.warning(
            "⚠️ **Kullanım Koşulları**\n\n"
            "Bu uygulama **yapay zeka destekli genel ilk yardım bilgi asistanıdır**.\n\n"
            "- Tıbbi tanı koymaz, tedavi önermez, reçete yazmaz.\n"
            "- Verilen bilgiler genel rehber niteliğindedir ve sağlık profesyoneli tavsiyesinin yerini **almaz**.\n"
            "- Acil durumlarda bu uygulamayı beklemeden **derhal 112'yi arayın**.\n"
            "- Kişisel sağlık verileriniz kaydedilmez; sohbet yalnızca oturum boyunca saklanır.\n\n"
            "Devam ederek bu koşulları kabul etmiş sayılırsınız."
        )
        accepted = st.checkbox("✅ Yukarıdaki koşulları okudum ve kabul ediyorum", key="consent_cb")
        
    if accepted:
        st.session_state.accepted_terms = True
        placeholder.empty()  # Sil ve yeniden yükleme (rerun) yapmadan devam et
        return False
        
    return True  # Henüz onay verilmedi — stop


# ─── Sidebar ───────────────────────────────────────────────────────────────────

SIDEBAR_QUICK_QUERIES = [
    "CPR nasıl yapılır?",
    "Yanıkta ne yapmalıyım?",
    "Burun kanamasını nasıl durdururum?",
    "Epilepsi nöbetinde ne yapılır?",
]


def render_sidebar():
    """Sidebar bileşenleri: stats, bilgi tabanı, hızlı aramalar, temizle."""
    with st.sidebar:
        st.markdown('<div class="sb-title">🚑 HayatKurtaran AI</div>', unsafe_allow_html=True)
        st.caption("RAG Mimarisi · Gemini Flash · FAISS · v3.0")
        st.divider()

        # İstatistikler
        st.markdown("**📊 Oturum İstatistikleri**")
        total_q = st.session_state.get("total_queries", 0)
        crit_c = st.session_state.get("critical_count", 0)
        st.markdown(
            f'<div class="stat-box">Toplam Soru: <strong>{total_q}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-box">Acil Durum Tespiti: <strong>{crit_c}</strong></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**📁 Bilgi Tabanı**")
        st.markdown("🩺 **İlk Yardım Bilgileri** — 30+ senaryo")
        st.markdown("🚨 **Acil Durumlar** — 17 bölüm, 160+ chunk")
        st.markdown("💊 **Sağlık Önerileri** — 37 konu başlığı")
        st.divider()

        st.markdown("**⚡ Hızlı Aramalar**")
        for i, q in enumerate(SIDEBAR_QUICK_QUERIES):
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
            'HayatKurtaran AI v3.0 · Senior Developer Edition<br>'
            'Veriler doğruluk açısından kontrol edilmiştir.</div>',
            unsafe_allow_html=True,
        )


def render_input_tools():
    """ChatGPT/Antigravity tarzı milimetrik birleşik giriş barı için modern katman."""
    from ui.styles import get_unified_input_css
    st.markdown(get_unified_input_css(), unsafe_allow_html=True)
    
    # ─── 1. Görsel (Plus) Tetikleyici ───
    with st.container():
        st.markdown('<div class="plus-btn-trigger">', unsafe_allow_html=True)
        if "show_uploader" not in st.session_state:
            st.session_state.show_uploader = False
            
        if st.button("➕", key="pp_plus_btn", help="Görsel Ekle"):
            st.session_state.show_uploader = not st.session_state.show_uploader
        st.markdown('</div>', unsafe_allow_html=True)
            
    # ─── 2. Ses (Mikrofon) Katmanı ───
    voice_query = None
    st.markdown('<div class="mic-btn-trigger">', unsafe_allow_html=True)
    from audio_recorder_streamlit import audio_recorder
    audio_bytes = audio_recorder(
        text="",
        recording_color="#ff4757",
        neutral_color="#94a3b8",
        icon_size="2.5x", # Biraz daha büyük, görsele uygun
        key="pp_mic"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if audio_bytes:
        from backend.audio_processor import transcribe_audio
        with st.spinner("Sesiniz metne çevriliyor..."):
            voice_query = transcribe_audio(audio_bytes)
            if voice_query:
                st.toast(f"🎤 Algılandı: {voice_query}", icon="✅")
                    
    # ─── 3. Görsel Yükleyici & HEMEN ÜSTTE Önizleme ───
    if st.session_state.show_uploader:
        st.markdown('<div class="vision-preview-container">', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="vision-preview-card">', unsafe_allow_html=True)
            u_file = st.file_uploader("Fotoğraf Seçin", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="pp_uploader")
            if u_file:
                from PIL import Image
                img = Image.open(u_file)
                st.session_state.active_image = img
                st.image(img, caption="Analize Hazır Görüntü", width=110)
                if st.button("✅ Tamam", key="pp_close_up"):
                    st.session_state.show_uploader = False
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
    return st.session_state.get("active_image"), voice_query


# ─── Quick Actions ─────────────────────────────────────────────────────────────

QUICK_ACTIONS = [
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


def render_quick_actions():
    """Hızlı aksiyon butonları (3 sütunlu grid)."""
    st.markdown(
        '<div class="quick-label">⚡ Hızlı Aksiyon — Acil Senaryolar</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for idx, (label, query) in enumerate(QUICK_ACTIONS):
        with cols[idx % 3]:
            if st.button(label, key=f"qa_{idx}", use_container_width=True):
                st.session_state.pending_query = query
    st.divider()


# ─── Emergency Buttons ────────────────────────────────────────────────────────

def render_emergency_buttons():
    st.markdown(get_emergency_button_css(), unsafe_allow_html=True)
    st.markdown(
        f'<div class="emergency-call-btn">'
        f'<a href="tel:112">🚨 112 ACİL ARAMA</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Akıllı GPS Butonları (Hastane + SMS)
    components.html(
        '''
        <style>
        .em-btn {
            background: linear-gradient(135deg, #0984e3, #74b9ff);
            color: white;
            font-family: sans-serif;
            font-weight: bold;
            font-size: 16px;
            padding: 14px 24px;
            border-radius: 12px;
            text-align: center;
            display: block;
            margin: 12px auto;
            width: 80%;
            max-width: 400px;
            cursor: pointer;
            border: none;
            box-shadow: 0 4px 10px rgba(9, 132, 227, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .em-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(9, 132, 227, 0.6); }
        .em-btn:active { transform: scale(0.98); }
        .hastane-btn { background: linear-gradient(135deg, #00b894, #55efc4); box-shadow: 0 4px 10px rgba(0, 184, 148, 0.4); }
        .hastane-btn:hover { box-shadow: 0 6px 15px rgba(0, 184, 148, 0.6); }
        </style>
        
        <button class="em-btn hastane-btn" onclick="openHospital()">🏥 EN YAKIN HASTANEYİ BUL</button>
        <button class="em-btn" onclick="sendEmergencyLocation()">📍 Konumumu 112'ye SMS Gönder</button>
        
        <script>
        function openHospital() {
            // Popup blocker engeline takılmamak için ilk tıklamada map açılır
            var newWindow = window.open("https://www.google.com/maps/search/hastane/", '_blank');
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    var lat = position.coords.latitude;
                    var lon = position.coords.longitude;
                    // Konum bulununca harita güncellenir
                    newWindow.location.href = "https://www.google.com/maps/search/hastane/@" + lat + "," + lon + ",15z";
                }, function(error) {}, { timeout: 7000 });
            }
        }
        function sendEmergencyLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    var lat = position.coords.latitude;
                    var lon = position.coords.longitude;
                    var mapsLink = "https://maps.google.com/?q=" + lat + "," + lon;
                    var msg = "ACİL DURUM! Lütfen bana yardım edin. Konumum: " + mapsLink;
                    window.parent.location.href = "sms:112?body=" + encodeURIComponent(msg);
                }, function(error) {
                    var msg = "ACİL DURUM! Cihaz konum bilgilerimi alamadım, lütfen benimle iletişime geçin.";
                    window.parent.location.href = "sms:112?body=" + encodeURIComponent(msg);
                }, { timeout: 7000 });
            } else {
                alert("Tarayıcınız konum özelliğini desteklemiyor.");
            }
        }
        </script>
        ''',
        height=160,
    )


# ─── Chat Response ────────────────────────────────────────────────────────────

def render_source_tags(sources: list, max_display: int = 3):
    """Kaynak etiketlerini render eder."""
    if not sources:
        return
    src_html = " ".join(
        f'<span class="src-tag">📄 {s["source"]}'
        f'{" > " + s["section"] if s.get("section") else ""}'
        f' ({int(s["score"]*100)}%)</span>'
        for s in sources[:max_display]
    )
    st.markdown(src_html, unsafe_allow_html=True)


def render_confidence_bar(confidence: float, has_context: bool):
    """Kaynak güvenilirlik progress bar'ı."""
    if has_context and confidence > 0:
        conf_pct = int(confidence * 100)
        conf_color = "🟢" if confidence >= 0.6 else "🟡" if confidence >= 0.45 else "🔴"
        st.progress(min(confidence, 1.0), text=f"{conf_color} Kaynak güvenilirliği: %{conf_pct}")


def render_latency_badge(latency_ms: float):
    """Yanıt süresini gösteren badge."""
    if latency_ms > 0:
        st.markdown(
            f'<span class="latency-badge">⏱️ Yanıt süresi: {latency_ms/1000:.1f}s</span>',
            unsafe_allow_html=True,
        )


def render_disclaimer():
    """Her yanıtın altında zorunlu yasal uyarı."""
    st.caption(
        "⚠️ **TIBBİ VE HUKUKİ UYARI:** Bu bilgi yapay zeka tarafından üretilmektedir. "
        "Sistem gönderdiğiniz **fotoğraflardan tıbbi teşhis koyamaz**; "
        "sadece genel ilk yardım rehberi sunar. Kesin tanı ve tedavi için "
        "derhal bir hekime başvurun veya 112'yi arayın."
    )


# ─── Welcome Message ──────────────────────────────────────────────────────────

def render_welcome():
    """İlk açılışta karşılama mesajı."""
    with st.chat_message("assistant", avatar="🚑"):
        st.markdown("""
Merhaba! Ben **HayatKurtaran AI** – acil ilk yardım ve sağlık konularında size Türkçe yardım eden bir yapay zekâ asistanıyım.

**Nasıl çalışırım?**
- Sorunuzu doğal dilde yazın: *"Arı soktu elimde şişlik var"*
- Ya da üstteki **Hızlı Aksiyon** butonlarından bir senaryo seçin

> 🚨 **Hayati tehlike durumunda bu uygulamayı beklemeyin — derhal 112'yi arayın!**
        """)
