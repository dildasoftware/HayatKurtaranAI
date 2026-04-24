# -*- coding: utf-8 -*-
"""
backend/voice.py — Text-to-Speech (TTS) Engine
==============================================
Yapay zeka yanıtlarını sese çevirerek elleri meşgul olan kullanıcılara yardımcı olur.
"""

import os
import time
from gtts import gTTS
import streamlit as st

# Ses dosyalarının geçici olarak saklanacağı klasör
TEMP_VOICE_DIR = os.path.join(os.getcwd(), "temp_audio")
if not os.path.exists(TEMP_VOICE_DIR):
    os.makedirs(TEMP_VOICE_DIR)


def text_to_speech(text: str, lang: str = "tr") -> str:
    """
    Metni sese çevirir ve dosya yolunu döner.
    Performans için sadece son 150 karakteri veya ilk anlamlı blokları seslendirebiliriz,
    ancak şimdilik tam metin üzerinden gidelim.
    """
    try:
        # Markdown etiketlerini ve emoji karakterlerini temizle (TTS'i bozmasın)
        clean_text = text.replace("**", "").replace("*", "").replace("#", "")
        # Linkleri temizle
        import re
        clean_text = re.sub(r'http\S+', '', clean_text)
        
        filename = f"voice_{int(time.time())}.mp3"
        filepath = os.path.join(TEMP_VOICE_DIR, filename)
        
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        tts.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"[Voice] TTS Hatası: {e}")
        return ""


def clean_old_voices():
    """Eski ses dosyalarını temizleyerek disk dolmasını önler."""
    try:
        now = time.time()
        for f in os.listdir(TEMP_VOICE_DIR):
            f_path = os.path.join(TEMP_VOICE_DIR, f)
            # 5 dakikadan eski dosyaları sil
            if os.stat(f_path).st_mtime < now - 300:
                os.remove(f_path)
    except Exception as e:
        print(f"[Voice] Temizlik Hatası: {e}")
