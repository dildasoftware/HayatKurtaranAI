"""
backend/config.py
Uygulama genelinde kullanılan temel konfigürasyon değerleri.

Ortam değişkenleri ile override edilebilecek sabitler burada tutulur.
"""

from __future__ import annotations

import os
from typing import Final, List


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_csv(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# ─── Vektör Veritabanı / Embedding Ayarları ─────────────────────────────────────

VECTOR_MODEL_NAME: Final[str] = os.getenv(
    "VECTOR_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

VECTOR_TOP_K: Final[int] = _get_int("VECTOR_TOP_K", 20)
VECTOR_SIMILARITY_THRESHOLD: Final[float] = _get_float(
    "VECTOR_SIMILARITY_THRESHOLD",
    0.05,
)

CRITICAL_KEYWORDS: Final[List[str]] = _get_csv(
    "CRITICAL_KEYWORDS",
    [
        "kalp krizi",
        "kalp durdu",
        "nefes almıyor",
        "nefes durdu",
        "bilinç kaybı",
        "bilincini kaybetti",
        "bayıldı",
        "ölüyor",
        "şiddetli kanama",
        "çok kan",
        "boğuluyor",
        "boğulma",
        "anafilaksi",
        "alerjik şok",
        "inme",
        "felç",
    ],
)


# ─── Gemini / LLM Ayarları ──────────────────────────────────────────────────────

GEMINI_MODEL_CANDIDATES: Final[List[str]] = _get_csv(
    "GEMINI_MODEL_CANDIDATES",
    [
        "gemini-2.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite-preview",
    ],
)

GEMINI_MAX_RETRIES: Final[int] = _get_int("GEMINI_MAX_RETRIES", 2)
GEMINI_RETRY_WAIT_SECONDS: Final[int] = _get_int("GEMINI_RETRY_WAIT_SECONDS", 1)

