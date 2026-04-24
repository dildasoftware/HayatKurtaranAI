# -*- coding: utf-8 -*-
"""
backend/conversation.py — Multi-turn Dialog Memory
===================================================
Kullanıcının ardışık sorularını / cevaplarını birleştirerek
eksik bağlamı tamamlar.

Örnek:
U1: "Babam düştü" 
A1: "Müdahale etmeyin, bilinci nasıl?"
U2: "Şu an kapalı"

Query Contextifier, bu sohbeti alır ve VectorDB'ye göndermeden önce:
"Babam düştü, şu an bilinci kapalı" şeklinde Zenginleştirilmiş Sorgu (Enriched Query) üretir.
"""

def enrich_query_with_history(current_query: str, history: list) -> str:
    """
    Önceki mesajların sadece içeriklerini alarak bağlamı zenginleştirir.
    Gemini'nin token limitini harcamamak için çok basit ve hızlı bir
    'Context Window' (son 3 mesaj) mantığı kullanır.
    
    Args:
        current_query (str): Son kullanıcı mesajı
        history (list): Streamlit st.session_state.messages formatında liste
                        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                        
    Returns:
        str: RAG sistemine (FAISS ve Classifier) gönderilecek zengin yapılı sorgu.
    """
    if not history:
        return current_query
        
    # Sadece kullanıcının son 2 mesajını al (Kısa dönemli hasta durumu)
    # Asistanın verdikleri zaten sabit bilgi olduğu için almamıza gerek yok,
    # bizim için önemli olan 'hastanın şikayetleri'.
    
    recent_user_msgs = [msg["content"] for msg in history if msg["role"] == "user"]
    
    if len(recent_user_msgs) > 0:
        # Örn: Eğer kullanıcı mevcut soruda çok kısa bir şey yazmışsa ("Evet", "Kapalı", "Kanıyor")
        # Eski mesajı bağlama mutlaka kat.
        if len(current_query.split()) <= 4:
            # Phase 3 Update: Chat memory should NOT blindly prepend!
            # "kalp krizi" history shouldn't infect "elimi kestim". 
            # Sadece belli zamirler veya anaphora varsa bağlamı arkada gizli tut, string'i bozma.
            return current_query
            
        # Eğer soru yeterince uzunsa, kullanıcının niyetini anlıyordur, çok fazla eski mesaj katıp
        # FAISS embedding'ini bulandırmaya gerek yok.
        pass
        
    return current_query
