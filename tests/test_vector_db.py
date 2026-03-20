import types

from backend import vector_db


def test_is_critical_query_detects_obvious_emergency():
    assert vector_db.is_critical_query("Hasta nefes almıyor, ne yapmalıyım?")
    assert vector_db.is_critical_query("Şiddetli kanama var, çok kan gidiyor")


def test_is_critical_query_ignores_non_emergency():
    assert not vector_db.is_critical_query("Hafif baş ağrım var, ne önerirsin?")


def test_get_context_returns_empty_when_no_chunks(monkeypatch):
    # Simüle: initialize hiç chunk yükleyememiş
    monkeypatch.setattr(vector_db, "_chunks", [])
    monkeypatch.setattr(vector_db, "_index", None)
    monkeypatch.setattr(vector_db, "initialize", lambda: None)

    ctx, sources, is_critical = vector_db.get_context("Test sorgu")

    assert ctx == ""
    assert sources == []
    assert is_critical is False


def test_get_context_with_fake_index(monkeypatch):
    # Küçük, sahte chunk listesi
    fake_chunks = [
        {
            "text": "Kalp krizi belirtileri ve CPR adimlari",
            "source": "ilk_yardim",
            "title": "Kalp Krizi",
            "metadata": {"filename": "fake.txt", "chunk_index": 0},
        },
        {
            "text": "Yanıkta ilk yardım uygulamalari",
            "source": "ilk_yardim",
            "title": "Yanık",
            "metadata": {"filename": "fake.txt", "chunk_index": 1},
        },
    ]

    class FakeIndex:
        def __init__(self) -> None:
            self._called = False

        def search(self, _emb, k):
            # Her zaman ilk chunk'i en yuksek skorla dondur
            import numpy as np

            self._called = True
            scores = np.array([[0.99] + [0.0] * (k - 1)], dtype="float32")
            indices = np.array([[0] + [-1] * (k - 1)], dtype="int64")
            return scores, indices

    def fake_load_model():
        # encode metoduna sahip minimal sahte model
        m = types.SimpleNamespace()

        def encode(texts, normalize_embeddings=True):
            import numpy as np

            # Tek boyutlu sahte embedding
            return np.ones((len(texts), 4), dtype="float32")

        m.encode = encode
        return m

    monkeypatch.setattr(vector_db, "_chunks", fake_chunks)
    monkeypatch.setattr(vector_db, "_index", FakeIndex())
    monkeypatch.setattr(vector_db, "_load_model", fake_load_model)

    ctx, sources, is_critical = vector_db.get_context("Kalp krizi nedir?")

    assert "Kalp Krizi" in ctx
    assert sources
    assert sources[0]["title"] == "Kalp Krizi"
    assert is_critical is True

