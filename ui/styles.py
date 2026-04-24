# -*- coding: utf-8 -*-
"""
ui/styles.py — HayatKurtaran AI CSS Stilleri
==============================================
Tüm Streamlit özel CSS tanımları bu modülde merkezi olarak yönetilir.
"""


def get_main_css() -> str:
    """Ana uygulama CSS stillerini döndürür."""
    return """
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

    /* ─── Latency badge ─── */
    .latency-badge {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.7rem;
        color: #8892a4;
        margin-top: 4px;
    }
    </style>
    """


def get_emergency_button_css() -> str:
    """Acil çağrı butonu CSS stillerini döndürür."""
    return """
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
    """


def get_panic_mode_css() -> str:
    """Acil durum panik modu (Seviye 1-2) teması CSS."""
    return """
    <style>
    .stApp {
        border-top: 6px solid #ff4757;
        border-bottom: 6px solid #ff4757;
        animation: panicPulse 2s ease-in-out infinite;
        box-shadow: inset 0 0 80px rgba(255,71,87,0.3);
    }
    @keyframes panicPulse {
        0%, 100% { border-color: #ff4757; box-shadow: inset 0 0 80px rgba(255,71,87,0.3); }
        50%      { border-color: #c0392b; box-shadow: inset 0 0 150px rgba(255,71,87,0.5); }
    }
    /* Chat girişini kırmızıyla vurgula */
    [data-testid="stChatInput"] {
        border-color: #ff4757 !important;
        box-shadow: 0 0 10px rgba(255,71,87,0.3);
    }
    </style>
    """

def get_unified_input_css() -> str:
    """ChatGPT/Antigravity tarzı milimetrik birleşik giriş barı için CSS."""
    return """
    <style>
    /* Chat girişini sabitle ve modernleştir */
    [data-testid="stChatInput"] {
        padding-left: 60px !important;
        padding-right: 60px !important;
        border-radius: 28px !important;
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 71, 87, 0.3) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        backdrop-filter: blur(12px);
    }
    
    /* Plus butonu tam sol iç konum */
    .plus-btn-trigger {
        position: fixed;
        bottom: 34px;
        left: calc(50% - 370px);
        z-index: 1001;
    }
    @media (max-width: 800px) {
        .plus-btn-trigger { left: 40px; }
    }
    
    /* Mic butonu tam sağ iç konum (Gönder butonunun solunda) */
    .mic-btn-trigger {
        position: fixed;
        bottom: 25px;
        right: calc(50% - 310px);
        z-index: 1001;
    }
    @media (max-width: 800px) {
        .mic-btn-trigger { right: 80px; }
    }
    
    /* Görsel önizleme - Yazı kutusunun HEMEN ÜZERİNDE */
    .vision-preview-container {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 740px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding-bottom: 5px;
    }
    
    .vision-preview-card {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(255, 71, 87, 0.5);
        border-radius: 14px;
        padding: 8px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        display: inline-block;
        animation: floatUp 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    @keyframes floatUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Streamlit butonlarını şeffaflaştır (sadece ikon kalsın) */
    .plus-btn-trigger .stButton button, .mic-btn-trigger .stButton button {
        background: transparent !important;
        border: none !important;
        color: #ff4757 !important;
        font-size: 24px !important;
        padding: 0 !important;
        min-height: unset !important;
        height: auto !important;
    }
    .plus-btn-trigger .stButton button:hover, .mic-btn-trigger .stButton button:hover {
        color: #fff !important;
        transform: scale(1.15);
    }
    </style>
    """
