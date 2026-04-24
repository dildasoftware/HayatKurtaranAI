# -*- coding: utf-8 -*-
"""
backend/faithfulness.py — LLM Yanıt Doğrulama (Rag-as-a-Judge)
===============================================================
RAG pipeline'dan çıkan yanıtın, gerçekten FAISS'den dönen bağlama
sadık kalıp kalmadığını (Faithfulness) ölçer.

Halüsinasyonu sıfıra indirmek için "Post-Generation Check" yapar.
Eğer LLM bağlamda olmayan bir tıbbi iddia üretmişse uyarır.
"""

import re
import numpy as np

# Embedding modelini semantic_classifier veya vector_db'den ortak kullanıyoruz
# Kaynak israfını önlemek için vector_db'nin modelini çağıracağız
from backend.vector_db import _load_model


def _split_into_claims(text: str) -> list[str]:
    """Yanıtı kontrol edilebilir küçük iddialara / cümlelere böler."""
    # Basit cümle sonlandırma karakterlerinden böl
    # Madde imlerini ve numaraları temizle
    clean_text = re.sub(r'[\*\-\>]', '', text)
    sentences = re.split(r'[.!?]\s+', clean_text)
    
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    return claims


def check_faithfulness(llm_answer: str, context_text: str, severity: int) -> dict:
    """
    LLM yanıtındaki cümleleri bağlam (context) metni ile karşılaştırarak
    söylenen her şeyin bağlamda olup olmadığını kontrol eder.
    
    Args:
        llm_answer: LLM'in ürettiği nihai yanıt
        context_text: FAISS'ten gelen destekleyici metin
        severity: Triyaj seviyesi (Kritik durumlarda eşik daha yüksektir)
        
    Returns:
        dict: sadakat skoru ve varsa doğrulanmamış iddialar.
    """
    if not llm_answer or not context_text:
        return {
            "is_faithful": False, 
            "score": 0.0, 
            "unverified_claims": []
        }
    
    claims = _split_into_claims(llm_answer)
    if not claims:
        return {
            "is_faithful": True, 
            "score": 1.0, 
            "unverified_claims": []
        }
        
    # Kritik durumlarda tolerans yok (threshold yüksek)
    threshold = 0.40 if severity <= 2 else 0.30
    
    model = _load_model()
    
    # Context'i embedding'e çevir (Paragraflara bölerek)
    context_chunks = [c.strip() for c in context_text.split('\n\n') if len(c.strip()) > 10]
    if not context_chunks:
        context_chunks = [context_text]
        
    ctx_embs = model.encode(context_chunks, convert_to_numpy=True)
    ctx_embs = ctx_embs / np.linalg.norm(ctx_embs, axis=1, keepdims=True)
    
    # İddiaları embedding'e çevir
    claim_embs = model.encode(claims, convert_to_numpy=True)
    claim_embs = claim_embs / np.linalg.norm(claim_embs, axis=1, keepdims=True)
    
    # Cosine Similarity Matrix
    similarities = np.dot(claim_embs, ctx_embs.T)
    
    unverified_claims = []
    total_score = 0
    
    for i, claim in enumerate(claims):
        max_sim = np.max(similarities[i])
        total_score += max_sim
        
        if max_sim < threshold:
            unverified_claims.append({
                "claim": claim,
                "confidence": float(max_sim)
            })
            
    avg_score = total_score / len(claims)
    is_faithful = len(unverified_claims) / len(claims) < 0.5  # %50'den fazlası uydurmaysa başarısız
    
    return {
        "is_faithful": is_faithful,
        "score": float(avg_score),
        "unverified_claims": unverified_claims
    }


def get_hallucination_warning_html() -> str:
    return (
        '<div style="background-color:rgba(255, 152, 0, 0.15); border-left:4px solid #ff9800; '
        'padding:10px; margin-top:10px; border-radius:4px; font-size:0.85rem; color:#ffb74d;">'
        '<strong>⚠️ Otomatik Doğrulama Uyarı:</strong> Bu yanıt, sistemin veritabanında '
        'doğrudan bulunmayan bazı çıkarımlar içermektedir. Doğruluğu şüpheli olabilir.'
        '</div>'
    )
