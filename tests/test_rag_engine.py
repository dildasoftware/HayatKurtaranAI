from backend import rag_engine


class DummyContext:
    def __init__(self, text: str) -> None:
        self.text = text


def test_generate_answer_no_context(monkeypatch):
    def fake_get_context(_query: str):
        return "", [], False

    monkeypatch.setattr(rag_engine, "get_context", fake_get_context)
    monkeypatch.setattr(rag_engine, "_initialized", True)

    result = rag_engine.generate_answer("Belirsiz bir soru")
    assert result["has_context"] is False
    assert "Bilgim dışında" in result["answer"]


def test_generate_answer_quota_error(monkeypatch):
    def fake_get_context(_query: str):
        return "baglam", [], True

    def fake_call_gemini(_prompt: str):
        return None, "quota"

    monkeypatch.setattr(rag_engine, "get_context", fake_get_context)
    monkeypatch.setattr(rag_engine, "_call_gemini", fake_call_gemini)
    monkeypatch.setattr(rag_engine, "_initialized", True)

    result = rag_engine.generate_answer("Kritik bir soru")
    answer = result["answer"]
    # Kritik uyarı + kota mesajı birlikte gelmeli
    assert "KRITIK ACIL DURUM ALGILANDI" in answer
    assert "API Kotası Doldu" in answer or "kota" in answer.lower()


def test_generate_answer_generic_error_fallback(monkeypatch):
    def fake_get_context(_query: str):
        return "Ilk yardim baglami", [], False

    def fake_call_gemini(_prompt: str):
        return None, "other"

    monkeypatch.setattr(rag_engine, "get_context", fake_get_context)
    monkeypatch.setattr(rag_engine, "_call_gemini", fake_call_gemini)
    monkeypatch.setattr(rag_engine, "_initialized", True)

    result = rag_engine.generate_answer("Soru")
    answer = result["answer"]
    assert "SİSTEM UYARISI" in answer
    assert "Ilk yardim baglami" in answer

