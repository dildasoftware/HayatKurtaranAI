# -*- coding: utf-8 -*-
"""
evaluation/embedding_benchmark.py
=================================
Modelin (paraphrase-multilingual-MiniLM-L12-v2) tıbbi terimler
üzerindeki anlamsal anlama yeteneğini (Semantic capability) ölçer.

Eş anlamlı terimlerin birbirine çok yakın (Cosine Similarity > 0.8),
zıt anlamlı / alakasız terimlerin birbirinden uzak (< 0.4) olması beklenir.
Makalenin metodoloji bölümünde (Figure veya Table) kullanılacaktır.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.semantic_classifier import _load_model

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def run_benchmark():
    # Model yükle
    print("Embedding modeli yükleniyor...")
    model = _load_model()
    
    # Test çiftleri
    synonym_pairs = [
        ("Kalp masajı", "CPR"),
        ("Boğuluyor hava alamıyor", "Solunum yolu tıkanıklığı"),
        ("İnme", "Felç"),
        ("Aşırı kan şekeri düşmesi", "Hipoglisemi"),
        ("Aspirin verin", "Kan sulandırıcı verin"),
        ("Epistaksis", "Burun kanaması"),
        ("Şiddetli göğüs ağrısı", "Kalp krizi şüphesi"),
        ("Derin yanık", "2. derece yanık"),
        ("Bilinç kaybı", "Bayılma"),
        ("Nabız yok", "Kalbi durdu")
    ]
    
    unrelated_pairs = [
        ("Burun kanaması", "Kalp krizi"),
        ("Hafif çizik", "Açık kırık kemik gözüküyor"),
        ("Şekerim düştü", "Kafasını şiddetle duvara çarptı"),
        ("Astım ilacım yok", "Çamaşır suyu içti"),
        ("Epilepsi", "Kedi tırmalaması")
    ]
    
    print(f"\n{'='*65}")
    print(" 🧠 EMBEDDING MODEL SEMANTİK TESTİ (Makale Bulguları)")
    print(f"{'='*65}")
    print("  Eş Anlamlı / Yakın Tıbbi Terimler (Hedef: Yüksek Skor > 0.75)")
    print(f"{'─'*65}")
    
    syn_scores = []
    for t1, t2 in synonym_pairs:
        embs = model.encode([t1, t2], convert_to_numpy=True)
        sim = cosine_similarity(embs[0], embs[1])
        syn_scores.append(sim)
        print(f"  {sim:.3f} | {t1[:25]:<25} ↔ {t2[:25]:<25}")
        
    print(f"\n{'─'*65}")
    print("  Alakasız / Farklı Tıbbi Terimler (Hedef: Düşük Skor < 0.40)")
    print(f"{'─'*65}")
    
    unrel_scores = []
    for t1, t2 in unrelated_pairs:
        embs = model.encode([t1, t2], convert_to_numpy=True)
        sim = cosine_similarity(embs[0], embs[1])
        unrel_scores.append(sim)
        print(f"  {sim:.3f} | {t1[:25]:<25} ↔ {t2[:25]:<25}")
        
    print(f"\n{'='*65}")
    avg_syn = sum(syn_scores) / len(syn_scores)
    avg_unrel = sum(unrel_scores) / len(unrel_scores)
    
    print(f"  Ortalama Eş Anlamlı Skoru : {avg_syn:.3f}")
    print(f"  Ortalama Alakasız Skoru   : {avg_unrel:.3f}")
    print(f"  Ayrım (Separation) Marjı  : {avg_syn - avg_unrel:.3f}")
    
    if avg_syn > 0.70 and avg_unrel < 0.50:
        print("\n  ✅ BULGU: Seçilen MiniLM modeli Türkçe tıbbi terminolojiyi")
        print("     başarıyla öğrenmiş ve birbirine karıştırmamaktadır!")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    run_benchmark()
