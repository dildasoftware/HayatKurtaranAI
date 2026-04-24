# -*- coding: utf-8 -*-
"""
backend/emergency_classifier.py — HayatKurtaran AI Acil Durum Sınıflandırıcısı
================================================================================
Kullanıcı sorgularını 5 seviyeli acil durum triyajına göre sınıflandırır.

Seviyeler (ESI — Emergency Severity Index benzeri):
  1 - KRİTİK:     Hayati tehlike, derhal 112 (kalp krizi, nefes durması, bilinç kaybı)
  2 - ACİL:        Hızlı müdahale gerekli (ciddi kanama, kırık, anafilaksi belirtileri)
  3 - ÖNCELİKLİ:  Yakın takip (orta yanık, astım atağı, hipoglisemi)
  4 - GENEL:       Ev bakımı uygulanabilir (sıyrık, hafif yanık, burun kanaması)
  5 - BİLGİ:       Koruyucu sağlık bilgisi (beslenme, egzersiz, stres)

Mimari:
  Bu versiyon keyword + pattern matching yaklaşımı kullanır (v1).
  Gelecekte BERT fine-tune modeline geçiş planlanmaktadır (v2 — makale novelty).
  
  Makale için karşılaştırma:
    - v1 (keyword-based) vs v2 (BERT-based) accuracy karşılaştırması
    - Precision/Recall/F1 metrikleri
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
from backend.semantic_classifier import classify_semantic


@dataclass
class TriageResult:
    """Triyaj sonucu."""
    severity: int           # 1-5 arası
    label: str              # Türkçe etiket
    confidence: float       # 0.0-1.0 arası güven skoru
    reasoning: str          # Neden bu seviye seçildi
    matched_patterns: list  # Hangi pattern'lar eşleşti


# ─── Seviye Tanımları ──────────────────────────────────────────────────────────
SEVERITY_LEVELS = {
    1: "🔴 KRİTİK — Hayati tehlike, DERHAL 112",
    2: "🟠 ACİL — Hızlı müdahale gerekli",
    3: "🟡 ÖNCELİKLİ — Yakın takip gerekli",
    4: "🟢 GENEL — Ev bakımı uygulanabilir",
    5: "🔵 BİLGİ — Koruyucu sağlık bilgisi",
}

# ─── Pattern Veritabanı ───────────────────────────────────────────────────────
# Her pattern: (regex_pattern, severity_level, confidence, description)
# severity 1'den 5'e doğru taranır; ilk eşleşen kazanır.

TRIAGE_PATTERNS = [
    # ═══════════════════════════════════════════════════════════════════
    # SEVİYE 1 — KRİTİK (hayati tehlike)
    # ═══════════════════════════════════════════════════════════════════
    (r"kalp\s*(krizi|durdu|durması|durmas)", 1, 0.95, "Kardiyak arrest şüphesi"),
    (r"kalbi\s*(durdu|durmuş|çalışmıyor)", 1, 0.95, "Kardiyak arrest"),
    (r"nefes\s*(almıyor|durdu|durmuş|yok)", 1, 0.95, "Solunum arresti"),
    (r"nefes\s*alam[ıi]yorum", 1, 0.90, "Akut solunum sıkıntısı"),
    (r"bilinc(ini)?\s*(kaybetti|kapandı|yok|kapalı)", 1, 0.95, "Bilinç kaybı"),
    (r"(ölüyor|öldü|ölmek\s*üzere)", 1, 0.95, "Hayati tehlike bildirimi"),
    (r"(felç|inme)\s*(geçiriyor|belirtileri|şüphesi)", 1, 0.90, "Akut serebrovasküler olay"),
    (r"(yüzü|yüzüm)\s*(düştü|eğildi|sarktı)", 1, 0.90, "Felç belirtisi (FAST)"),
    (r"(kolu|kolum)\s*(kalkm[ıi]yor|tutam[ıi]yor)", 1, 0.90, "Felç belirtisi (FAST)"),
    (r"konuş(amıyor|ması\s*bozuk|muyor)", 1, 0.85, "Felç/bilinç değişikliği"),
    (r"göğs(üm|ün)de\s*(ağrı|baskı|sıkışma|ezilme)", 1, 0.90, "Akut koroner sendrom şüphesi"),
    (r"göğüs\s*ağrısı", 1, 0.85, "Göğüs ağrısı — kardiyak dışlanmalı"),
    (r"anafilak(si|tik)", 1, 0.95, "Anafilaktik şok"),
    (r"boğaz[ıi](m)?\s*şiş(iyor|ti|miş)", 1, 0.90, "Üst hava yolu obstrüksiyonu"),
    (r"dil(im)?\s*şiş(iyor|ti|miş)", 1, 0.90, "Anjioödem"),
    (r"(bebek|çocuk)\s*(nefes\s*almıyor|morar)", 1, 0.95, "Pediatrik solunum arresti"),
    (r"suya\s*düştü.*çıkar(amıyor|amadık)", 1, 0.90, "Aktif boğulma"),
    (r"elektrik\s*çarptı", 1, 0.85, "Elektrik yaralanması"),

    # ═══════════════════════════════════════════════════════════════════
    # SEVİYE 2 — ACİL (hızlı müdahale)
    # ═══════════════════════════════════════════════════════════════════
    (r"(çok|şiddetli|durmuyor)\s*kan(ama|ıyor)?", 2, 0.85, "Ciddi kanama"),
    (r"kan\s*(fışkırıyor|durmuyor|çok\s*fazla)", 2, 0.90, "Kontrol edilemeyen kanama"),
    (r"(yüksekten|merdi[vn]en|çatı)\s*(düştü|düşme)", 2, 0.85, "Yüksekten düşme travması"),
    (r"kafa\s*(travması|yaralanması|çarptı)", 2, 0.85, "Kafa travması"),
    (r"(kafasını|başını)\s*çarptı", 2, 0.80, "Kafa travması"),
    (r"(yılan|akrep)\s*(ısır|sok)", 2, 0.90, "Zehirli hayvan yaralanması"),
    (r"(zehir|ilaç)\s*(içti|yuttu|aldı)", 2, 0.90, "Zehirlenme"),
    (r"hapları\s*yuttu", 2, 0.90, "İlaç intoksikasyonu"),
    (r"havale\s*geçiriyor", 2, 0.85, "Konvülziyon"),
    (r"nöbet\s*(geçiriyor|5\s*dk|süren)", 2, 0.85, "Status epilepticus şüphesi"),
    (r"(geniş|büyük|ciddi)\s*yanık", 2, 0.85, "Ciddi yanık"),
    (r"(yüz|el|genital)\s*yanık", 2, 0.85, "Kritik bölge yanığı"),
    (r"kimyasal\s*(yanık|madde|temas)", 2, 0.85, "Kimyasal yaralanma"),
    (r"(açık|kemik\s*görünen)\s*kırık", 2, 0.90, "Açık kırık"),
    (r"boğul(uyor|du|ma)", 2, 0.85, "Boğulma riski"),
    (r"(cereyan|akım)a?\s*kapıldı", 2, 0.85, "Elektrik yaralanması"),
    (r"(karbon\s*monoksit|co\s*zehir)", 2, 0.90, "CO zehirlenmesi"),
    (r"morar(ıyor|mış|dı).*dudak", 2, 0.85, "Siyanoz belirtisi"),

    # ═══════════════════════════════════════════════════════════════════
    # SEVİYE 3 — ÖNCELİKLİ (yakın takip)
    # ═══════════════════════════════════════════════════════════════════
    (r"(orta|2\.\s*derece)\s*yanık", 3, 0.80, "Parsiyel kalınlık yanığı"),
    (r"astım\s*(krizi|atağı|nefes)", 3, 0.80, "Astım alevlenmesi"),
    (r"(şeker|kan\s*şekeri)\s*(düştü|düşük|düşmesi)", 3, 0.85, "Hipoglisemi"),
    (r"şekeri\s*(düştü|düşük)", 3, 0.85, "Hipoglisemi"),
    (r"titr(eme|iyor).*terle(me|yor)", 3, 0.75, "Hipoglisemi belirtileri"),
    (r"diyabet.*bilinç", 3, 0.85, "Diyabetik acil"),
    (r"tansiyon\s*(çok\s*yüksek|kriz|180|200)", 3, 0.80, "Hipertansif kriz"),
    (r"(kırık|kırıl|kırıldı)", 3, 0.75, "Kırık şüphesi"),
    (r"çıkık", 3, 0.75, "Eklem çıkığı"),
    (r"bayıl(dı|ıyor|ma)", 3, 0.75, "Senkop"),
    (r"arı\s*(soktu|ısırdı|sokması)", 3, 0.70, "Arı sokması — anafilaksi izle"),
    (r"(böcek|kene)\s*(sok|ısır)", 3, 0.70, "Böcek/kene yaralanması"),
    (r"(derin|uzun)\s*kesik", 3, 0.75, "Derin laserasyon"),
    (r"nöbet\s*(geçirdi|bitti|sonrası)", 3, 0.75, "Postnöbet değerlendirme"),
    (r"ateş.*39|39.*ateş|40.*ateş|ateş.*40", 3, 0.75, "Yüksek ateş"),

    # ═══════════════════════════════════════════════════════════════════
    # SEVİYE 4 — GENEL (ev bakımı)
    # ═══════════════════════════════════════════════════════════════════
    (r"(hafif|küçük|ufak)\s*(yanık|kesik|yara)", 4, 0.75, "Minör yaralanma"),
    (r"(sıyrık|çizik|tırmık)", 4, 0.80, "Yüzeysel yaralanma"),
    (r"burun\s*kan(aması|ıyor)", 4, 0.75, "Epistaksis"),
    (r"burkulma", 4, 0.80, "Burkulma"),
    (r"burkul(du|muş|dum)", 4, 0.80, "Burkulma"),
    (r"(ayağım|bileğim|dizim)\s*(burkul|incin|ağrıyor)", 4, 0.75, "Eklem yaralanması"),
    (r"(baş\s*ağrısı|başım\s*ağrıyor)", 4, 0.65, "Baş ağrısı — red flag taranmalı"),
    (r"mide\s*(bulantı|bulanıyor)", 4, 0.70, "Bulantı"),
    (r"(ishal|kusma|kustu)", 4, 0.70, "GİS şikayeti"),
    (r"(öksürük|öksürüyor|boğaz\s*ağrısı)", 4, 0.65, "Üst solunum yolu"),
    (r"(morluk|çürük|bere)", 4, 0.75, "Yumuşak doku travması"),
    (r"güneş\s*yanığı", 4, 0.75, "Güneş yanığı"),
    (r"(kıymık|battan|iğne)", 4, 0.80, "Yabancı cisim — yüzeysel"),
    (r"(kes(ik|tim|ildi|ti)|yara(landı|lı))", 4, 0.65, "Kesik/yara — derinliğe göre değerlendir"),
    (r"(ateş|sıcaklık).*37|38.*ateş", 4, 0.60, "Subfebril ateş"),

    # ═══════════════════════════════════════════════════════════════════
    # SEVİYE 5 — BİLGİ (koruyucu sağlık)
    # ═══════════════════════════════════════════════════════════════════
    (r"(nasıl\s*korunur|önlem|önleme)", 5, 0.80, "Koruyucu sağlık bilgisi"),
    (r"(beslenme|diyet|vitamin|mineral)", 5, 0.80, "Beslenme bilgisi"),
    (r"(egzersiz|spor|koşu|yürüyüş)", 5, 0.75, "Fiziksel aktivite"),
    (r"(stres|uyku|meditasyon|rahatlama)", 5, 0.75, "Mental sağlık"),
    (r"(ilkyardım\s*çantası|malzeme|kit)", 5, 0.80, "İlkyardım hazırlığı"),
    (r"ilkyardım\s*(nedir|tanımı|eğitimi)", 5, 0.85, "İlkyardım genel bilgi"),
    (r"(aşı|bağışıklık|hijyen)", 5, 0.75, "Koruyucu sağlık"),
]


def classify(query: str, use_hybrid: bool = True) -> TriageResult:
    """
    Kullanıcı sorgusunu acil durum seviyesine göre sınıflandırır.
    
    Args:
        query: Kullanıcının sorduğu soru
        use_hybrid: Eğer True ise, v1 (regex) ve v2 (semantik) ensemble yapar.
        
    Returns:
        TriageResult: Triyaj sonucu
    """
    query_lower = query.lower().strip()
    
    # ─── V1: REGEX SINIFLANDIRICI ───────────────────────────────────────────
    v1_severity = 5
    v1_confidence = 0.3
    v1_reasoning = "Belirli bir acil durum pattern'ı tespit edilemedi"
    matched = []
    
    for pattern, severity, confidence, description in TRIAGE_PATTERNS:
        if re.search(pattern, query_lower):
            matched.append({
                "pattern": pattern, "severity": severity,
                "confidence": confidence, "description": description,
            })
            if severity < v1_severity or (severity == v1_severity and confidence > v1_confidence):
                v1_severity = severity
                v1_confidence = confidence
                v1_reasoning = description
    
    if len(matched) > 1:
        v1_confidence = min(v1_confidence + 0.05 * (len(matched) - 1), 0.99)
        
    # Eğer hibrid istenmiyorsa veya semantik devredışıysa düz regex dön
    if not use_hybrid:
        return TriageResult(
            severity=v1_severity,
            label=SEVERITY_LEVELS.get(v1_severity, "BİLGİ"),
            confidence=v1_confidence,
            reasoning=f"[v1] {v1_reasoning}",
            matched_patterns=matched,
        )

    # ─── V2: SEMANTİK SINIFLANDIRICI (K-NN) ─────────────────────────────────
    v2_result = classify_semantic(query)
    v2_severity = v2_result["severity"]
    v2_confidence = v2_result["confidence"]
    v2_reasoning = v2_result["reasoning"]
    
    # ─── ENSEMBLE (GÜVENLİK ÖNCELİKLİ) ──────────────────────────────────────
    best_severity = 5
    best_confidence = 0.0
    best_reasoning = ""
    
    # Kural: Hangisi hastayı daha "kritik" değerlendirmişse ve güveni yüksekse onu al
    # v1 ve v2 aynı fikirdeyse
    if v1_severity == v2_severity:
        best_severity = v1_severity
        best_confidence = min(max(v1_confidence, v2_confidence) + 0.1, 0.99) # Güven artar
        best_reasoning = f"[Ensemble-Agree] {v1_reasoning} | {v2_reasoning}"
    # Farklı fikirdeyse, "düşük numara" yani DAHA KRİTİK olanı seç (Güvenlik Önceliği)
    else:
        if v1_severity < v2_severity:
            best_severity = v1_severity
            best_confidence = v1_confidence
            best_reasoning = f"[Ensemble-V1] {v1_reasoning}"
        else:
            best_severity = v2_severity
            best_confidence = v2_confidence
            best_reasoning = f"[Ensemble-V2] {v2_reasoning}"
            
    # --- PHASE 3: SAFETY GUARDRAIL ---
    # Eğer sistem hiçbir şey bulamayıp Seviye 5 dediyse, ama cümlenin içinde fiziksel yaralanma
    # veya travma belirtileri varsa otonom olarak Seviye 2'ye zorla.
    if best_severity > 3:
        danger_keywords = ["kan", "yara", "kes", "kop", "vur", "çarp", "düş", "kır", "patla", "şiş", "pas"]
        if any(dk in query_lower for dk in danger_keywords):
            best_severity = 2
            best_confidence = 0.85
            best_reasoning = f"[Safety-Guardrail] Seviye 5'ten Seviye 2'ye çekildi (Kritik kelime tespiti)"
            
    return TriageResult(
        severity=best_severity,
        label=SEVERITY_LEVELS.get(best_severity, "BİLGİ"),
        confidence=best_confidence,
        reasoning=best_reasoning,
        matched_patterns=matched,
    )


def get_severity_badge_html(result: TriageResult) -> str:
    """Triyaj sonucunu HTML badge olarak döndürür (Streamlit UI için)."""
    colors = {
        1: "#ff1744",  # Kırmızı
        2: "#ff6d00",  # Turuncu
        3: "#ffd600",  # Sarı
        4: "#00c853",  # Yeşil
        5: "#2979ff",  # Mavi
    }
    color = colors.get(result.severity, "#9e9e9e")
    
    return (
        f'<div style="background:{color}; color:white; padding:8px 16px; '
        f'border-radius:8px; display:inline-block; font-weight:bold; '
        f'font-size:14px; margin:4px 0;">'
        f'Triyaj: {result.label} (güven: %{int(result.confidence*100)})'
        f'</div>'
    )


# ─── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Babam kalp krizi geçiriyor ne yapmalıyım",
        "Göğsümde ağrı var baskı hissediyorum",
        "Çocuk boğuluyor nefes alamıyor",
        "Felç geçiriyor yüzü düştü konuşamıyor",
        "Kolundan çok kan geliyor durmuyor",
        "Yüksekten düştü kafasını çarptı",
        "Yılan ısırdı ne yapmalıyım",
        "Zehir içti çocuk",
        "Astım krizi geçiriyor inhaler yok",
        "Şekeri düştü titremeye başladı",
        "Arı soktu şişti",
        "Ayağım burkuldu",
        "Hafif bir kesik var",
        "Baş ağrısı var ne yapabilirim",
        "İlkyardım çantasında neler olmalı",
        "Nasıl kilo veririm",
    ]
    
    for q in test_queries:
        result = classify(q)
        print(f"Sev:{result.severity} | Güven:{result.confidence:.0%} | "
              f"{result.reasoning[:40]:40s} | {q}")
