# -*- coding: utf-8 -*-
"""
evaluation/run_evaluation.py — HayatKurtaran AI RAG Evaluation
================================================================
Nicel RAG performans metrikleri hesaplama.

Metrikler:
  - Context Precision  : Döndürülen chunk'lar gerçekten ilgili mi?
  - Keyword Hit Rate   : Beklenen anahtar kelimeler chunk'larda var mı?
  - OOS Rejection Rate : Alakasız soruları doğru reddediyor mu?
  - Source Accuracy     : Doğru dosyadan chunk döndürüyor mu?
  - Avg. Confidence     : Ortalama benzerlik skoru
  - Category Breakdown  : Kategori bazlı performans

Kullanım:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --export results.json
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.vector_db import get_context, initialize  # noqa: E402


# ─── Veri Yükleme ─────────────────────────────────────────────────────────────

def load_dataset(path: str = None) -> list[dict]:
    """Test veri setini yükle."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Metrik Hesaplama ──────────────────────────────────────────────────────────

def evaluate_single(item: dict) -> dict:
    """Tek bir sorgu için RAG performansını değerlendir."""
    query = item["query"]
    expected_keywords = item.get("expected_keywords", [])
    expected_source = item.get("expected_source", "")
    is_oos = item.get("category") == "out_of_scope"

    start = time.perf_counter()
    context_text, sources, is_critical, avg_confidence = get_context(query)
    latency_ms = (time.perf_counter() - start) * 1000

    # Keyword Hit Rate — beklenen kelimelerin kaçı context'te bulunuyor?
    keyword_hits = 0
    if expected_keywords and context_text:
        ctx_lower = context_text.lower()
        for kw in expected_keywords:
            if kw.lower() in ctx_lower:
                keyword_hits += 1
    keyword_hit_rate = keyword_hits / max(len(expected_keywords), 1) if expected_keywords else None

    # Source Accuracy — ilk sonuç doğru dosyadan mı geliyor?
    source_correct = False
    if sources and expected_source and expected_source != "none":
        for s in sources[:3]:
            if expected_source.lower() in s.get("source", "").lower():
                source_correct = True
                break

    # OOS Rejection — out-of-scope soruda boş veya düşük güvenli sonuç dönmeli
    oos_rejected = None
    if is_oos:
        # OOS başarılı = sonuç sayısı <= 1 VEYA confidence < 0.45
        oos_rejected = len(sources) <= 1 or avg_confidence < 0.45

    return {
        "id": item["id"],
        "query": query,
        "category": item.get("category", ""),
        "difficulty": item.get("difficulty", ""),
        "source_count": len(sources),
        "avg_confidence": round(avg_confidence, 4),
        "keyword_hit_rate": round(keyword_hit_rate, 4) if keyword_hit_rate is not None else None,
        "source_correct": source_correct,
        "oos_rejected": oos_rejected,
        "is_critical": is_critical,
        "latency_ms": round(latency_ms, 2),
    }


# ─── Toplu Değerlendirme ──────────────────────────────────────────────────────

def run_full_evaluation(dataset: list[dict]) -> dict:
    """Tüm veri seti üzerinde değerlendirme çalıştır."""
    print(f"\n{'='*70}")
    print(f"  HayatKurtaran AI — RAG Evaluation Report")
    print(f"  {len(dataset)} sorgu üzerinde değerlendirme başlıyor...")
    print(f"{'='*70}\n")

    results = []
    for i, item in enumerate(dataset, 1):
        result = evaluate_single(item)
        results.append(result)
        status = "✓" if (result.get("keyword_hit_rate") is None or result["keyword_hit_rate"] > 0.5) else "✗"
        print(f"  [{i:3d}/{len(dataset)}] {status} {item['id']:6s} | "
              f"conf={result['avg_confidence']:.2f} | "
              f"kw={result['keyword_hit_rate']:.0%}" if result['keyword_hit_rate'] is not None
              else f"  [{i:3d}/{len(dataset)}] {status} {item['id']:6s} | "
              f"conf={result['avg_confidence']:.2f} | OOS",
              end="")
        if result['latency_ms'] > 100:
            print(f" | {result['latency_ms']:.0f}ms ⚠️")
        else:
            print(f" | {result['latency_ms']:.0f}ms")

    # ─── Genel Metrikler ───────────────────────────────────────────────
    medical_results = [r for r in results if r["keyword_hit_rate"] is not None]
    oos_results = [r for r in results if r["oos_rejected"] is not None]

    # Keyword Hit Rate
    kw_rates = [r["keyword_hit_rate"] for r in medical_results]
    avg_kw_hit = sum(kw_rates) / max(len(kw_rates), 1)
    perfect_kw = sum(1 for r in kw_rates if r >= 1.0) / max(len(kw_rates), 1)
    partial_kw = sum(1 for r in kw_rates if 0.5 <= r < 1.0) / max(len(kw_rates), 1)
    miss_kw = sum(1 for r in kw_rates if r < 0.5) / max(len(kw_rates), 1)

    # Source Accuracy
    source_acc = sum(1 for r in medical_results if r["source_correct"]) / max(len(medical_results), 1)

    # OOS Rejection Rate
    oos_rejection_rate = sum(1 for r in oos_results if r["oos_rejected"]) / max(len(oos_results), 1)

    # Confidence
    all_conf = [r["avg_confidence"] for r in results]
    med_conf = [r["avg_confidence"] for r in medical_results]

    # Latency
    latencies = [r["latency_ms"] for r in results]
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]

    # ─── Kategori Bazlı ───────────────────────────────────────────────
    cat_stats = defaultdict(lambda: {"count": 0, "kw_sum": 0, "conf_sum": 0, "src_correct": 0})
    for r in medical_results:
        cat = r["category"]
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["kw_sum"] += r["keyword_hit_rate"]
        cat_stats[cat]["conf_sum"] += r["avg_confidence"]
        if r["source_correct"]:
            cat_stats[cat]["src_correct"] += 1

    # ─── Rapor Yazdır ──────────────────────────────────────────────────
    report = {
        "total_queries": len(results),
        "medical_queries": len(medical_results),
        "oos_queries": len(oos_results),
        "keyword_hit_rate": round(avg_kw_hit, 4),
        "perfect_keyword_match": round(perfect_kw, 4),
        "partial_keyword_match": round(partial_kw, 4),
        "keyword_miss_rate": round(miss_kw, 4),
        "source_accuracy": round(source_acc, 4),
        "oos_rejection_rate": round(oos_rejection_rate, 4),
        "avg_confidence_all": round(sum(all_conf) / max(len(all_conf), 1), 4),
        "avg_confidence_medical": round(sum(med_conf) / max(len(med_conf), 1), 4),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_avg_ms": round(sum(latencies) / max(len(latencies), 1), 2),
    }

    print(f"\n{'='*70}")
    print(f"  📊 GENEL SONUÇLAR")
    print(f"{'='*70}")
    print(f"  Toplam Sorgu           : {report['total_queries']}")
    print(f"  Tıbbi Sorgu            : {report['medical_queries']}")
    print(f"  Out-of-Scope           : {report['oos_queries']}")
    print(f"{'─'*70}")
    print(f"  ✅ Keyword Hit Rate    : {report['keyword_hit_rate']:.1%}")
    print(f"     Tam Eşleşme         : {report['perfect_keyword_match']:.1%}")
    print(f"     Kısmi Eşleşme       : {report['partial_keyword_match']:.1%}")
    print(f"     Kaçırılan           : {report['keyword_miss_rate']:.1%}")
    print(f"  ✅ Source Accuracy     : {report['source_accuracy']:.1%}")
    print(f"  ✅ OOS Rejection Rate  : {report['oos_rejection_rate']:.1%}")
    print(f"{'─'*70}")
    print(f"  📈 Confidence (tıbbi)  : {report['avg_confidence_medical']:.3f}")
    print(f"  📈 Confidence (tümü)   : {report['avg_confidence_all']:.3f}")
    print(f"{'─'*70}")
    print(f"  ⏱️  Latency P50       : {report['latency_p50_ms']:.1f} ms")
    print(f"  ⏱️  Latency P95       : {report['latency_p95_ms']:.1f} ms")
    print(f"  ⏱️  Latency Avg       : {report['latency_avg_ms']:.1f} ms")

    print(f"\n{'='*70}")
    print(f"  📋 KATEGORİ BAZLI SONUÇLAR")
    print(f"{'='*70}")
    print(f"  {'Kategori':<20} {'Sayı':>5} {'KW Hit':>8} {'Conf':>8} {'Src':>6}")
    print(f"  {'─'*52}")
    for cat, stats in sorted(cat_stats.items()):
        n = stats["count"]
        kw = stats["kw_sum"] / n
        conf = stats["conf_sum"] / n
        src = stats["src_correct"] / n
        print(f"  {cat:<20} {n:>5} {kw:>7.1%} {conf:>7.3f} {src:>5.0%}")

    print(f"{'='*70}\n")

    report["category_breakdown"] = {
        cat: {
            "count": s["count"],
            "keyword_hit_rate": round(s["kw_sum"] / s["count"], 4),
            "avg_confidence": round(s["conf_sum"] / s["count"], 4),
            "source_accuracy": round(s["src_correct"] / s["count"], 4),
        }
        for cat, s in cat_stats.items()
    }
    report["detailed_results"] = results

    return report


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HayatKurtaran AI RAG Evaluation")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--dataset", type=str, help="Custom dataset path")
    args = parser.parse_args()

    # Sistem başlat
    print("[Eval] Sistem başlatılıyor...")
    initialize()

    # Veri yükle
    dataset = load_dataset(args.dataset)
    print(f"[Eval] {len(dataset)} sorgu yüklendi")

    # Değerlendirme çalıştır
    report = run_full_evaluation(dataset)

    # Export
    if args.export:
        export_path = args.export
    else:
        export_path = os.path.join(os.path.dirname(__file__), "rag_results.json")

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Eval] Sonuçlar kaydedildi: {export_path}")
