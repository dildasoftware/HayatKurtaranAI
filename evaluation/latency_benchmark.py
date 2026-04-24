# -*- coding: utf-8 -*-
"""
evaluation/latency_benchmark.py — Latency Ölçümü
=================================================
Gerçek senaryoda (FAISS + Gemini API) end-to-end süreyi ölçer.

Makale için sistem performansının (gecikme) canlı durumda
ölçülmesini sağlar. Maliyet ve rate-limit nedeniyle sadece
rastgele seçilen N sorgu üzerinde çalışır.
"""

import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag_engine import generate_answer, startup


def load_sample_queries(n: int = 15) -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Sadece tıbbi konulardan rastgele seç
    medical = [item["query"] for item in data if item["category"] != "out_of_scope"]
    random.seed(42)  # Tekrar edilebilirlik için sabit seed
    return random.sample(medical, min(n, len(medical)))


def run_latency_benchmark():
    startup()
    queries = load_sample_queries(15)  # API kotasını bitirmemek için 15 yeterli
    
    print(f"\n{'='*60}")
    print(f" ⏱️ END-TO-END LATENCY BENCHMARK (N={len(queries)})")
    print(f"{'='*60}")
    
    latencies = []
    
    for i, q in enumerate(queries, 1):
        print(f"[{i:2d}/{len(queries)}] {q[:35]:<35}...", end="", flush=True)
        
        # İlk sorgu genelde daha yavaştır (warmup), ama gerçekçi olması için katıyoruz
        start = time.perf_counter()
        res = generate_answer(q)
        total_ms = (time.perf_counter() - start) * 1000
        
        # Engine içindeki ayrıştırılmış süreleri al (logger ölçtüğü için objeden çekemiyoruz doğrudan)
        # Ama dict içinde 'latency_ms' dönüyor
        latency_val = res.get("latency_ms", total_ms)
        latencies.append(latency_val)
        
        print(f" {latency_val:>6.0f} ms")
        
        # Rate limit yememek için kısa bekleme
        time.sleep(1.5)
        
    latencies.sort()
    avg_ms = sum(latencies) / len(latencies)
    p50_ms = latencies[len(latencies) // 2]
    p95_ms = latencies[int(len(latencies) * 0.95)]
    
    print(f"\n{'='*60}")
    print(f"  📊 LATENCY SONUÇLARI (LLM Network Dahil)")
    print(f"{'='*60}")
    print(f"  Ortalama Süre    : {avg_ms/1000:.2f} saniye")
    print(f"  Medyan (P50)     : {p50_ms/1000:.2f} saniye")
    print(f"  P95 (Kötü Senaryo): {p95_ms/1000:.2f} saniye")
    print(f"  En Hızlı         : {latencies[0]/1000:.2f} saniye")
    print(f"  En Yavaş         : {latencies[-1]/1000:.2f} saniye")
    print(f"{'='*60}\n")
    
    # Makale için veriyi kaydet
    out_path = os.path.join(os.path.dirname(__file__), "latency_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": len(queries),
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "raw_latencies": latencies
        }, f, indent=2)


if __name__ == "__main__":
    run_latency_benchmark()
