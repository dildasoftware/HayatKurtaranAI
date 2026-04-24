# -*- coding: utf-8 -*-
"""
backend/semantic_classifier.py — Semantic Emergency Classifier (v2)
======================================================================
Makine öğrenmesi tabanlı, kelime anlamına bakan triyaj sınıflandırıcı.

Regex (v1) düz mantıkla çalışırken, bu modül (v2) cümlenin niyetini anlar.
"Sentence-Transformers + FAISS K-NN" mimarisiyle çok hızlı ve isabetli.
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Sabitler
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "evaluation", "triage_training_data.json")

# Global variables
_model = None
_index = None
_training_data = []


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index():
    """Eğitim verilerini okur ve K-NN yapısı için FAISS indeksi oluşturur."""
    global _index, _training_data
    
    if not os.path.exists(DATA_FILE):
        return
        
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        _training_data = json.load(f)
        
    model = _load_model()
    texts = [item["text"] for item in _training_data]
    
    # Embeddingleri hesapla
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # FAISS indeksi oluştur
    dimension = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dimension)
    _index.add(embeddings)
    

def classify_semantic(query: str, k: int = 3) -> dict:
    """
    Sorgunun vektörünü K-NN ile eğitim setiyle karşılaştırır.
    
    Returns:
        dict: {
            "severity": int,
            "confidence": float,
            "reasoning": str,
            "neighbors": list
        }
    """
    global _index, _training_data
    
    if _index is None:
        build_index()
        
    model = _load_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    
    scores, indices = _index.search(q_emb, k)
    
    # Komşuları ve ağırlıkları topla
    neighbors = []
    severity_weights = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    
    for i in range(k):
        idx = indices[0][i]
        score = float(scores[0][i])
        
        if idx < 0 or idx >= len(_training_data):
            continue
            
        neighbor = _training_data[idx]
        sev = neighbor["severity"]
        
        # Güven formülü: yakınlık skorunu ceza/ödül çarpanı ile ağırlıklandır
        # Sadece çok benzer olanlar dikkate alınır
        if score > 0.40:
            severity_weights[sev] += score
            neighbors.append({
                "text": neighbor["text"],
                "severity": sev,
                "score": score
            })
            
    if not neighbors:
        return {
            "severity": 5, 
            "confidence": 0.3, 
            "reasoning": "Semantik eşleşme bulunamadı",
            "neighbors": []
        }
        
    # En yüksek ağırlıklı sınıfı seç
    best_severity = max(severity_weights, key=severity_weights.get)
    max_weight = severity_weights[best_severity]
    
    # Güven hesaplama (max weight üzerinden logaritmik vb. olabilir, basit tuttuk)
    total_weight = sum(severity_weights.values())
    confidence = min((max_weight / total_weight) * neighbors[0]["score"] * 1.5, 0.99)
    
    return {
        "severity": best_severity,
        "confidence": confidence,
        "reasoning": f"Semantik benzerlik: '{neighbors[0]['text']}' (Skor: {neighbors[0]['score']:.2f})",
        "neighbors": neighbors
    }


if __name__ == "__main__":
    test_queries = [
        "Kolumu kestim, kan akıyor",
        "Gece nasıl uyurum",
        "Düşüp kafasını çarptı",
        "Penceremin camı kırıldı", # OOS
    ]
    for q in test_queries:
        res = classify_semantic(q)
        print(f"Q: {q[:30]:30s} | Sev: {res['severity']} | Conf: {res['confidence']:.2f} | Reason: {res['reasoning']}")
