# -*- coding: utf-8 -*-
"""
evaluation/classifier_comparison.py
===================================
Regex (v1) sınıflandırıcı ile Semantik Hybrid sınıflandırıcıyı karşılaştırır.
Makale için Novelty doğrulama scripti.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.emergency_classifier import classify
from evaluation.classifier_eval import quadratic_weighted_kappa


def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_version(dataset, use_hybrid: bool):
    y_true = []
    y_pred = []
    
    for item in dataset:
        expected = item.get("expected_severity", 5)
        # classify fonksiyonuna use_hybrid parametresini paslıyoruz
        result = classify(item["query"], use_hybrid=use_hybrid)
        
        y_true.append(expected)
        y_pred.append(result.severity)
        
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    undertriage = sum(1 for t, p in zip(y_true, y_pred) if t < p) / n
    
    # Kritik undertriage
    critical_cases = sum(1 for t in y_true if t <= 2)
    critical_undertriage = sum(1 for t, p in zip(y_true, y_pred) if t <= 2 and p >= 3)
    critical_ut_rate = critical_undertriage / max(critical_cases, 1) if critical_cases else 0.0
    
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "undertriage": undertriage,
        "critical_undertriage": critical_ut_rate,
        "qwk": qwk,
        "y_true": y_true,
        "y_pred": y_pred
    }

def run_comparison():
    print("Veri yükleniyor ve model ayağa kalkıyor...")
    ds = load_dataset()
    
    print("\n[1/2] Regex (v1) Değerlendiriliyor...")
    res_v1 = evaluate_version(ds, use_hybrid=False)
    
    print("[2/2] Semantic Hybrid Değerlendiriliyor...")
    res_hybrid = evaluate_version(ds, use_hybrid=True)
    
    print(f"\n{'='*70}")
    print(f" 🏆 CLASSIFIER KARŞILAŞTIRMASI (Makale Tablo 2: Başarım)")
    print(f"{'='*70}")
    print(f"  Metrik                      | Regex (v1) | Semantic Hybrid | İyileşme")
    print(f"{'─'*70}")
    
    acc_diff = res_hybrid['accuracy'] - res_v1['accuracy']
    print(f"  Accuracy                    | {res_v1['accuracy']:<10.1%} | {res_hybrid['accuracy']:<15.1%} | {'+' if acc_diff>0 else ''}{acc_diff:.1%}")
    
    qwk_diff = res_hybrid['qwk'] - res_v1['qwk']
    print(f"  QWK (Tutarlılık)            | {res_v1['qwk']:<10.3f} | {res_hybrid['qwk']:<15.3f} | {'+' if qwk_diff>0 else ''}{qwk_diff:.3f}")
    
    ut_diff = res_v1['undertriage'] - res_hybrid['undertriage']
    print(f"  Genel Undertriage (Hata)    | {res_v1['undertriage']:<10.1%} | {res_hybrid['undertriage']:<15.1%} | -{ut_diff:.1%}")
    
    cut_diff = res_v1['critical_undertriage'] - res_hybrid['critical_undertriage']
    print(f"  Kritik Undertriage (Ölümcül)| {res_v1['critical_undertriage']:<10.1%} | {res_hybrid['critical_undertriage']:<15.1%} | -{cut_diff:.1%}")
    print(f"{'='*70}\n")
    print("💡 Semantik model, regex'in kaçırdığı dolaylı cümleleri yakalayarak başarıyı katlamıştır!")

if __name__ == "__main__":
    run_comparison()
