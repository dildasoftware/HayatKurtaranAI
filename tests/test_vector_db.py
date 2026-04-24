# -*- coding: utf-8 -*-
"""
tests/test_vector_db.py — HayatKurtaran AI Vektör DB Testleri
==============================================================
Chunking, benzerlik eşiği, acil kelime tespiti ve query enhancement testleri.
Bu testler makale için doğruluk metriklerinin temelini oluşturur.
"""

import os
import sys
import pytest

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.chunker import chunk_file, chunk_directory, _split_with_overlap
from backend.vector_db import (
    is_critical_query,
    _enhance_query,
    initialize,
    get_context,
    SIMILARITY_THRESHOLD,
    TOP_K,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKER TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunker:
    """Smart Chunker modülü testleri."""

    def test_chunk_directory_returns_chunks(self):
        """Veri klasörü chunk'lanabilmeli."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        chunks = chunk_directory(data_dir)
        assert len(chunks) > 0, "Hiç chunk oluşturulmadı!"

    def test_chunk_count_reasonable(self):
        """Chunk sayısı makul aralıkta olmalı (50-500)."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        chunks = chunk_directory(data_dir)
        assert 50 <= len(chunks) <= 500, f"Chunk sayısı beklenmedik: {len(chunks)}"

    def test_chunk_has_required_fields(self):
        """Her chunk gerekli alanları içermeli."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        chunks = chunk_directory(data_dir)
        required_fields = ["text", "source", "section", "subsection", "title", "keywords", "metadata"]
        for chunk in chunks[:10]:
            for field in required_fields:
                assert field in chunk, f"Chunk'ta '{field}' alanı eksik"

    def test_chunk_text_not_empty(self):
        """Chunk metni boş olmamalı."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        chunks = chunk_directory(data_dir)
        for chunk in chunks:
            assert len(chunk["text"].strip()) > 0, "Boş chunk tespit edildi!"

    def test_chunk_size_reasonable(self):
        """Chunk boyutu makul aralıkta olmalı (40-1200 karakter)."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        chunks = chunk_directory(data_dir)
        for chunk in chunks:
            length = len(chunk["text"])
            assert length >= 40, f"Chunk çok kısa ({length} char)"
            assert length <= 1200, f"Chunk çok uzun ({length} char)"

    def test_section_context_injected(self):
        """Chunk metninin başında section context prefix olmalı."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        chunks = chunk_directory(data_dir)
        bracket_count = sum(1 for c in chunks if c["text"].startswith("["))
        # En az %80'inde prefix olmalı
        assert bracket_count / len(chunks) > 0.8, "Section context injection yetersiz"

    def test_overlap_function(self):
        """Overlap fonksiyonu doğru çalışmalı."""
        text = "A" * 1000
        pieces = _split_with_overlap(text, 400, 80)
        assert len(pieces) >= 2, "Uzun metin bölünmeli"
        # Her parça max_size'ı aşmamalı (azıcık sapma olabilir cümle sınırı nedeniyle)
        for p in pieces:
            assert len(p) <= 500, f"Parça çok büyük: {len(p)} char"


# ═══════════════════════════════════════════════════════════════════════════════
# ACİL KELİME TESPİTİ TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

class TestCriticalDetection:
    """Acil durum kelime tespiti testleri."""

    @pytest.mark.parametrize("query,expected", [
        # --- MUTLAKA TRUE OLMALI ---
        ("Kalp krizi geçiriyor", True),
        ("Babam nefes almıyor", True),
        ("Bilincini kaybetti yere düştü", True),
        ("Çok kan kaybediyor", True),
        ("Anafilaksi şoku geçiriyor", True),
        ("Felç geçiriyor konuşamıyor", True),
        ("Göğsümde ağrı var baskı hissediyorum", True),
        ("Nefes alamıyorum çok zor", True),
        ("Bebek nefes almıyor", True),
        ("Elektrik çarptı bilinci kapandı", True),
        ("Yılan ısırdı ne yapmalıyım", True),
        ("Çocuk boğuluyor acil", True),
        ("Suya düştü çıkaramıyoruz", True),
        ("Havale geçiriyor yere düştü", True),
        # --- MUTLAKA FALSE OLMALI ---
        ("Baş ağrısı var ne yapabilirim", False),
        ("Hapşırık tutamıyorum", False),
        ("Nasıl kilo veririm", False),
        ("Uykum gelmiyor ne yapmalıyım", False),
        ("Stresim var rahatlamak istiyorum", False),
    ])
    def test_critical_keyword_detection(self, query, expected):
        """Acil kelimeler doğru tespit edilmeli."""
        result = is_critical_query(query)
        assert result == expected, f"'{query}' -> beklenen={expected}, sonuç={result}"


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY ENHANCEMENT TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryEnhancement:
    """Sorgu zenginleştirme testleri."""

    def test_burn_query_enhanced(self):
        """Yanık ile ilgili sorgu zenginleştirilmeli."""
        result = _enhance_query("Elim yandı ne yapmalıyım")
        assert "yanık" in result.lower()

    def test_bleeding_query_enhanced(self):
        """Kanama ile ilgili sorgu zenginleştirilmeli."""
        result = _enhance_query("Parmağım kanıyor durmuyor")
        assert "kanama" in result.lower()

    def test_insect_query_enhanced(self):
        """Böcek/arı sokması sorgusu zenginleştirilmeli."""
        result = _enhance_query("Arı soktu ne yapayım")
        assert "anafilaksi" in result.lower() or "alerji" in result.lower()

    def test_plain_query_unchanged(self):
        """İlgisiz sorgu değiştirilmemeli."""
        original = "Bugün hava nasıl"
        result = _enhance_query(original)
        assert result == original


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS ARAMA TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

class TestFAISSSearch:
    """FAISS vektör arama entegrasyon testleri."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_db(self):
        """Veritabanını test başında bir kez başlat."""
        initialize()

    def test_relevant_query_returns_results(self):
        """İlgili sorgu sonuç döndürmeli."""
        ctx, sources, is_crit, conf = get_context("Yanıkta ilk yardım nasıl yapılır?")
        assert len(sources) > 0, "İlgili sorgu sonuç döndürmedi!"
        assert conf > 0.0

    def test_irrelevant_query_returns_empty(self):
        """Alakasız sorgu sonuç döndürmemeli (0.40 eşiği sayesinde)."""
        ctx, sources, is_crit, conf = get_context("Blockchain teknolojisi nedir?")
        # Eşik 0.40 olduğu için alakasız sorgu çok az veya hiç sonuç döndürmeli
        assert len(sources) <= 2, f"Alakasız sorgu {len(sources)} sonuç döndürdü — eşik çok düşük?"

    def test_critical_query_detected(self):
        """Kritik sorgu is_critical=True döndürmeli."""
        _, _, is_crit, _ = get_context("Kalp krizi geçiriyor ne yapmalıyım")
        assert is_crit is True

    def test_top_k_limit_respected(self):
        """Sonuç sayısı TOP_K'yı aşmamalı."""
        _, sources, _, _ = get_context("CPR nasıl yapılır?")
        assert len(sources) <= TOP_K, f"TOP_K={TOP_K} ama {len(sources)} sonuç döndü"

    def test_similarity_threshold_enforced(self):
        """Tüm dönen sonuçlar eşiğin üstünde olmalı."""
        _, sources, _, _ = get_context("Arı sokması ilk yardım")
        for src in sources:
            assert src["score"] >= SIMILARITY_THRESHOLD, (
                f"Eşik altı sonuç: {src['title']} (score={src['score']:.3f})"
            )

    def test_confidence_score_range(self):
        """Confidence score 0-1 aralığında olmalı."""
        _, _, _, conf = get_context("Burun kanaması nasıl durdurulur?")
        assert 0.0 <= conf <= 1.0, f"Confidence aralık dışı: {conf}"


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRİK DOĞRULUK TESTLERİ (Makale için baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccuracyBaseline:
    """Sistemin bilinen sorulara doğru chunk döndürmesi gerekir."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_db(self):
        initialize()

    @pytest.mark.parametrize("query,expected_keyword_in_context", [
        ("CPR nasıl yapılır?", "kompresyon"),
        ("Heimlich manevrası nasıl uygulanır?", "heimlich"),
        ("Yanıkta ne yapmalıyım?", "yanık"),
        ("Kalp krizi belirtileri nelerdir?", "göğüs"),
        ("Epilepsi nöbetinde ne yapılır?", "nöbet"),
        ("Arı sokmasında ilk yardım", "sokma"),
        ("Burun kanaması nasıl durur?", "burun"),
        ("Hipoglisemi ilk yardımı", "şeker"),
        ("Boğulma ilk yardımı", "solunum"),
        ("Kırıkta ilk yardım", "tespit"),  # kırık chunk'ı tespit/atel içerir
    ])
    def test_correct_context_retrieved(self, query, expected_keyword_in_context):
        """Bilinen sorular doğru bağlamı getirmeli."""
        ctx, sources, _, _ = get_context(query)
        assert expected_keyword_in_context.lower() in ctx.lower(), (
            f"'{query}' sorgusu '{expected_keyword_in_context}' kelimesini içeren chunk getirmedi!\n"
            f"Gelen context ilk 200 char: {ctx[:200]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
