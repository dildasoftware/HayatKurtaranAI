# -*- coding: utf-8 -*-
"""
backend/logger.py — HayatKurtaran AI Structured Logger
=======================================================
Her sorgu için yapısal log kaydı tutar (JSONL formatında).

Log alanları:
  - timestamp           : ISO 8601 zaman damgası
  - query               : Kullanıcı sorgusu
  - triage_severity     : Triyaj seviyesi (1-5)
  - triage_confidence   : Triyaj güven skoru
  - rag_confidence      : RAG ortalama benzerlik skoru
  - is_critical         : Acil durum tetiklendi mi?
  - has_context         : Bağlam bulundu mu?
  - source_count        : Dönen kaynak sayısı
  - latency_faiss_ms    : FAISS arama süresi (ms)
  - latency_llm_ms      : LLM API süresi (ms)
  - latency_total_ms    : Toplam end-to-end süre (ms)
  - faithfulness_score  : Post-gen doğrulama skoru (Faz 3'te eklenecek)

Kullanım:
    from backend.logger import QueryLogger
    logger = QueryLogger()
    logger.log({...})
    stats = logger.get_stats()
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ─── Konfigürasyon ─────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "queries.jsonl")
MAX_LOG_SIZE_MB = 50  # Log dosyası bu boyutu aşarsa rotate edilir


@dataclass
class QueryLog:
    """Tek bir sorgu logu."""
    timestamp: str = ""
    query: str = ""
    triage_severity: int = 5
    triage_confidence: float = 0.0
    rag_confidence: float = 0.0
    is_critical: bool = False
    has_context: bool = False
    source_count: int = 0
    latency_faiss_ms: float = 0.0
    latency_llm_ms: float = 0.0
    latency_total_ms: float = 0.0
    faithfulness_score: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class QueryLogger:
    """
    JSONL formatında yapısal sorgu logger'ı.
    Thread-safe değil (Streamlit single-thread, sorun olmaz).
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or LOG_FILE
        self._ensure_dir()

    def _ensure_dir(self):
        """Log dizinini oluştur."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def _rotate_if_needed(self):
        """Dosya çok büyükse rotate et."""
        try:
            if os.path.exists(self.log_file):
                size_mb = os.path.getsize(self.log_file) / (1024 * 1024)
                if size_mb > MAX_LOG_SIZE_MB:
                    rotated = self.log_file + f".{int(time.time())}.bak"
                    os.rename(self.log_file, rotated)
                    print(f"[Logger] Log rotated: {rotated}")
        except Exception:
            pass

    def log(self, entry: QueryLog):
        """Tek bir log girişi yaz."""
        self._rotate_if_needed()
        try:
            data = asdict(entry)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Logger] Yazma hatası: {e}")

    def get_recent(self, n: int = 50) -> list[dict]:
        """Son n log girişini döndür."""
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            recent = lines[-n:] if len(lines) > n else lines
            return [json.loads(line) for line in recent if line.strip()]
        except Exception:
            return []

    def get_stats(self) -> dict:
        """Tüm loglardan istatistik hesapla."""
        logs = self.get_recent(10000)  # Son 10K log
        if not logs:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "avg_confidence": 0,
                "critical_count": 0,
                "severity_distribution": {},
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
            }

        latencies = [l.get("latency_total_ms", 0) for l in logs if l.get("latency_total_ms", 0) > 0]
        confidences = [l.get("rag_confidence", 0) for l in logs if l.get("rag_confidence", 0) > 0]
        severities = [l.get("triage_severity", 5) for l in logs]

        # Severity dağılımı
        sev_dist = {}
        for s in severities:
            sev_dist[s] = sev_dist.get(s, 0) + 1

        # Percentile hesaplama
        sorted_lat = sorted(latencies) if latencies else [0]
        p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
        p95_idx = int(len(sorted_lat) * 0.95)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0

        return {
            "total_queries": len(logs),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 3),
            "critical_count": sum(1 for l in logs if l.get("is_critical")),
            "severity_distribution": sev_dist,
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
        }


# ─── Latency Timer ────────────────────────────────────────────────────────────

class LatencyTimer:
    """
    Context manager ile latency ölçümü.

    Kullanım:
        timer = LatencyTimer()
        with timer.measure("faiss"):
            # FAISS arama
        with timer.measure("llm"):
            # LLM çağrısı
        print(timer.results)  # {"faiss_ms": 42.3, "llm_ms": 1203.5, "total_ms": 1245.8}
    """

    def __init__(self):
        self._start_total = time.perf_counter()
        self._results: dict[str, float] = {}
        self._current_label: Optional[str] = None
        self._current_start: float = 0

    class _MeasureContext:
        def __init__(self, timer: "LatencyTimer", label: str):
            self.timer = timer
            self.label = label

        def __enter__(self):
            self.timer._current_start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = (time.perf_counter() - self.timer._current_start) * 1000
            self.timer._results[f"{self.label}_ms"] = round(elapsed, 2)

    def measure(self, label: str) -> _MeasureContext:
        """Belirli bir işlem bloğunun süresini ölç."""
        return self._MeasureContext(self, label)

    @property
    def results(self) -> dict[str, float]:
        """Tüm ölçüm sonuçlarını döndür (total_ms dahil)."""
        total = (time.perf_counter() - self._start_total) * 1000
        return {**self._results, "total_ms": round(total, 2)}


# ─── Singleton Logger ─────────────────────────────────────────────────────────
_logger: Optional[QueryLogger] = None


def get_logger() -> QueryLogger:
    """Tekil logger instance döndür."""
    global _logger
    if _logger is None:
        _logger = QueryLogger()
    return _logger
