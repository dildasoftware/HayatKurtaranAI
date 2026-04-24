# -*- coding: utf-8 -*-
"""
backend/audio_processor.py - Gemini Multimodal STT (Speech-to-Text)
"""
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Gemini multimodal yeteneğini kullanarak ses verisini metne çevirir.
    
    Args:
        audio_bytes (bytes): Web arayüzünden gelen ses verisi
        
    Returns:
        str: Transkript edilmiş metin
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return ""
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-flash-lite-latest")
        
        # Gemini 2.x audio bytes direct support
        # We wrap it in a multimodal content list
        prompt = "Bu ses kaydını tam olarak yazıya dök. SADECE metni döndür, ek açıklama yapma."
        
        # Multimodal request with audio
        response = model.generate_content([
            prompt,
            {
                "mime_type": "audio/wav",
                "data": audio_bytes
            }
        ])
        
        text = response.text.strip() if response.text else ""
        return text
        
    except Exception as e:
        print(f"[AudioSTT] Hata: {e}")
        # Fallback: Quota fail vs ise boş dön
        return ""
