# -*- coding: utf-8 -*-
"""
backend/chunker.py — HayatKurtaran AI Smart Chunker
====================================================
Medikal bilgi tabanı dosyalarını RAG için optimize edilmiş
chunk'lara bölen akıllı metin işleme modülü.

Özellikler:
  - ======== section separator tanıma
  - [TAG] alt başlık tanıma
  - Section context injection (her chunk hangi bölümden geldiğini bilir)
  - Overlap desteği (chunk sınırlarında bağlam kopması önlenir)
  - Zengin metadata (section, subsection, keywords, source)

Kaynak format örneği:
    ========================================
    BÖLÜM 2: TEMEL YAŞAM DESTEĞİ (KPR / CPR)
    ========================================

    [TANIM]
    Temel Yaşam Desteği...

    [YETİŞKİN CPR — ADIM ADIM]
    1. Ortam güvenliğini kontrol et.
    ...
"""

import os
import re
from typing import Optional


# ─── Konfigürasyon ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 600       # Karakter (hedef chunk boyutu)
CHUNK_OVERLAP = 80     # Karakter (chunk sınırlarında overlap)
MIN_CHUNK_LENGTH = 40  # Bu uzunluğun altındaki chunk'lar atlanır

# Section separator regex — ======== ve ━━━ satırlarını yakalar
SECTION_SEP_PATTERN = re.compile(
    r'={8,}\s*\n(.+?)\n\s*={8,}',
    re.DOTALL
)

# ## BÖLÜM X: başlık formatı (yeni acil_durumlar.txt)
BOLUM_HEADER_PATTERN = re.compile(
    r'^━{4,}\s*\n##\s+BÖLÜM\s+([A-Z]):?\s*(.+?)\s*\n━{4,}',
    re.MULTILINE
)

# ALL-CAPS satır başlıkları (saglik_onerileri.txt: "GÜNLÜK SU TÜKETİMİ VE HİDRASYON")
CAPSHEADER_PATTERN = re.compile(
    r'^([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s,İŞÖÜÇĞ()\-/]{10,})$',
    re.MULTILINE
)

# [TAG] alt başlık regex — hem [CPR ADIM] hem [A-01] formatını yakalar
SUBTAG_PATTERN = re.compile(
    r'^\[([A-ZÇĞİÖŞÜa-zçğıöşü0-9][A-ZÇĞİÖŞÜa-zçğıöşü\s/—\-\d:.,()]+)\]',
    re.MULTILINE
)


def _extract_keywords(text: str) -> list[str]:
    """Chunk metninden önemli anahtar kelimeleri çıkarır (basit TF yaklaşımı)."""
    # Medikal terimler ve önemli kelimeler
    medical_terms = [
        "CPR", "KPR", "AED", "Heimlich", "FAST", "BE-FAST", "RICE",
        "TYD", "ABC", "ABCDE", "GKS", "AVPU", "SBAR",
        "112", "114", "EpiPen", "nalokson", "atropin",
        "kompresyon", "ventilasyon", "defibrilatör", "ROSC",
        "kanama", "kırık", "yanık", "zehirlenme", "boğulma", "travma",
        "kalp krizi", "felç", "inme", "anafilaksi", "şok", "sepsis",
        "hipoglisemi", "epilepsi", "astım", "hipertansiyon", "DKA",
        "STEMI", "NSTEMI", "pnömotoraks", "hemotoraks", "emboli",
        "turnike", "bandaj", "atel", "sedye", "splint",
        "bebek", "çocuk", "yetişkin", "gebe", "yaşlı",
        "KBRN", "toksidrom", "antidot", "START", "SALT",
    ]
    found = []
    text_lower = text.lower()
    for term in medical_terms:
        if term.lower() in text_lower:
            found.append(term)
    return found[:8]  # En fazla 8 keyword


def _split_with_overlap(text: str, max_size: int, overlap: int) -> list[str]:
    """
    Uzun metni overlap ile parçalara böler.
    Cümle sınırlarını korumaya çalışır.
    """
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Cümle sonu arayarak doğal kırılma noktası bul
        best_break = end
        for sep in ['\n\n', '\n', '. ', '.\n', '; ', ', ']:
            pos = text.rfind(sep, start + max_size // 2, end + 50)
            if pos > start:
                best_break = pos + len(sep)
                break

        chunk_text = text[start:best_break].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Overlap: bir sonraki chunk, bu chunk'ın son kısmını da içerir
        start = max(start + 1, best_break - overlap)

    return chunks


def chunk_file(filepath: str, source_name: Optional[str] = None) -> list[dict]:
    """
    Tek bir dosyayı akıllı chunk'lara böler.

    Args:
        filepath: Dosyanın tam yolu
        source_name: Kaynak etiketi (varsayılan: dosya adı)

    Returns:
        list[dict]: Her biri şu alanları içeren chunk listesi:
            - text: Chunk metni (section context prefix dahil)
            - source: Kaynak dosya adı
            - section: Ana bölüm başlığı
            - subsection: Alt bölüm etiketi (varsa)
            - title: Chunk'ın ilk satırı (kısa başlık)
            - keywords: Önemli terimler listesi
            - metadata: {filename, chunk_index, section_index}
    """
    if source_name is None:
        source_name = os.path.splitext(os.path.basename(filepath))[0]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[Chunker] Dosya okunamadı: {filepath} — {e}")
        return []

    chunks = []

    # ─── Bölümlere ayır ────────────────────────────────────────────────────
    sections = []

    # Strateji 1: ======== section separator
    eq_parts = SECTION_SEP_PATTERN.split(content)
    if len(eq_parts) > 1:
        if eq_parts[0].strip():
            sections.append(("GENEL BİLGİ", eq_parts[0].strip()))
        for i in range(1, len(eq_parts) - 1, 2):
            section_title = eq_parts[i].strip()
            section_body = eq_parts[i + 1].strip() if i + 1 < len(eq_parts) else ""
            if section_body:
                sections.append((section_title, section_body))

    # Strateji 2: ━━━ + ## BÖLÜM X: formatı (yeni acil_durumlar.txt)
    if not sections:
        bolum_matches = list(BOLUM_HEADER_PATTERN.finditer(content))
        if bolum_matches:
            # Header (kaynak bilgisi vb.) — ilk bölümden önceki kısım
            header_text = content[:bolum_matches[0].start()].strip()
            if header_text:
                sections.append(("GENEL BİLGİ", header_text))
            for idx, m in enumerate(bolum_matches):
                letter = m.group(1)
                title = m.group(2).strip()
                section_title = f"BÖLÜM {letter}: {title}"
                start = m.end()
                end = bolum_matches[idx + 1].start() if idx + 1 < len(bolum_matches) else len(content)
                body = content[start:end].strip()
                # ━━━ satırlarını temizle
                body = re.sub(r'━{4,}', '', body).strip()
                if body:
                    sections.append((section_title, body))

    # Strateji 3: ALL-CAPS başlıklar (saglik_onerileri.txt)
    if not sections:
        caps_matches = list(CAPSHEADER_PATTERN.finditer(content))
        if caps_matches:
            for idx, m in enumerate(caps_matches):
                title = m.group(1).strip()
                start = m.end()
                end = caps_matches[idx + 1].start() if idx + 1 < len(caps_matches) else len(content)
                body = content[start:end].strip()
                if body:
                    sections.append((title, body))

    # Strateji 4: Fallback — tek bölüm
    if not sections:
        sections = [("GENEL", content.strip())]

    section_idx = 0
    chunk_global_idx = 0

    for section_title, section_body in sections:
        # ─── Alt bölümlere ayır ([TAG] bazlı) ─────────────────────────────
        # [TAG] pattern'ını bul ve alt bölümlere böl
        sub_parts = SUBTAG_PATTERN.split(section_body)

        sub_sections = []
        if sub_parts[0].strip():
            sub_sections.append((None, sub_parts[0].strip()))

        for j in range(1, len(sub_parts) - 1, 2):
            sub_tag = sub_parts[j].strip()
            sub_body = sub_parts[j + 1].strip() if j + 1 < len(sub_parts) else ""
            if sub_body:
                sub_sections.append((sub_tag, sub_body))

        # Eğer [TAG] yoksa, tüm body tek alt bölüm
        if not sub_sections:
            sub_sections = [(None, section_body)]

        for sub_tag, sub_body in sub_sections:
            # Section context prefix — her chunk'ın hangi bölümden geldiğini bilmesini sağlar
            context_prefix = f"[{section_title}]"
            if sub_tag:
                context_prefix += f" [{sub_tag}]"

            # Overlap ile parçalara böl
            text_pieces = _split_with_overlap(sub_body, CHUNK_SIZE, CHUNK_OVERLAP)

            for piece in text_pieces:
                if len(piece) < MIN_CHUNK_LENGTH:
                    continue

                # Chunk metni: context prefix + asıl metin
                chunk_text = f"{context_prefix}\n{piece}"

                # İlk satırı başlık olarak kullan
                first_line = piece.split('\n')[0].strip()
                if len(first_line) > 80:
                    first_line = first_line[:77] + "..."

                chunks.append({
                    "text": chunk_text,
                    "source": source_name,
                    "section": section_title,
                    "subsection": sub_tag or "",
                    "title": first_line,
                    "keywords": _extract_keywords(piece),
                    "metadata": {
                        "filename": os.path.basename(filepath),
                        "chunk_index": chunk_global_idx,
                        "section_index": section_idx,
                    }
                })
                chunk_global_idx += 1

        section_idx += 1

    print(f"[Chunker] {source_name}: {chunk_global_idx} chunk oluşturuldu "
          f"({len(sections)} bölüm)")
    return chunks


def chunk_directory(data_dir: str) -> list[dict]:
    """
    Bir klasördeki tüm .txt dosyalarını chunk'lar.

    Args:
        data_dir: Veri klasörü yolu

    Returns:
        list[dict]: Tüm dosyalardan toplanan chunk listesi
    """
    all_chunks = []

    if not os.path.exists(data_dir):
        print(f"[Chunker] UYARI: Veri dizini bulunamadı: {data_dir}")
        return all_chunks

    txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
    if not txt_files:
        print("[Chunker] UYARI: data/ klasöründe .txt dosyası bulunamadı.")
        return all_chunks

    for filename in txt_files:
        filepath = os.path.join(data_dir, filename)
        source_name = filename.replace(".txt", "")
        file_chunks = chunk_file(filepath, source_name)
        all_chunks.extend(file_chunks)

    print(f"[Chunker] TOPLAM: {len(all_chunks)} chunk ({len(txt_files)} dosya)")
    return all_chunks


# ─── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    chunks = chunk_directory(data_dir)

    print(f"\n{'='*60}")
    print(f"Toplam chunk sayısı: {len(chunks)}")
    print(f"Ortalama chunk uzunluğu: {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} karakter")
    print(f"{'='*60}")

    # İlk 3 chunk'ı göster
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Section: {c['section']}")
        print(f"Subsection: {c['subsection']}")
        print(f"Title: {c['title']}")
        print(f"Keywords: {c['keywords']}")
        print(f"Source: {c['source']}")
        print(f"Text ({len(c['text'])} char): {c['text'][:200]}...")
