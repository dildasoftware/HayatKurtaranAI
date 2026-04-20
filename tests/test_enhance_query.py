from backend.vector_db import _enhance_query


def test_enhance_query_adds_burn_keywords():
    q = "Elimi yaktım, ne yapmalıyım?"
    out = _enhance_query(q)
    assert out != q
    assert "yanık" in out.lower()


def test_enhance_query_handles_child_and_pregnancy():
    q_child = "Bebek düştü ne yapmalıyım?"
    out_child = _enhance_query(q_child)
    assert "pediatrik" in out_child.lower()

    q_preg = "Hamileyim tansiyonum yüksek"
    out_preg = _enhance_query(q_preg)
    assert "gebelik" in out_preg.lower()

