# -*- coding: utf-8 -*-
"""
evaluation/classifier_eval.py — Emergency Classifier Evaluation
================================================================
Triyaj sınıflandırıcısının performansını ölçer.

Metrikler:
  - Confusion Matrix (5×5)
  - Accuracy, Precision, Recall, F1 (macro + weighted)
  - Undertriage Rate (tehlikeli hata: kritik vakayı düşük sınıflandırma)
  - Overtriage Rate (kaynak israfı: hafif vakayı yüksek sınıflandırma)
  - Quadratic Weighted Kappa (QWK)

Kullanım:
    python evaluation/classifier_eval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.emergency_classifier import classify  # noqa: E402


def load_dataset(path: str = None) -> list[dict]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def quadratic_weighted_kappa(y_true: list, y_pred: list, num_classes: int = 5) -> float:
    """
    Quadratic Weighted Kappa (QWK) hesapla.
    Triage severity gibi ordinal sınıflandırmalar için standart metrik.
    """
    # Confusion matrix
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t - 1][p - 1] += 1

    n = len(y_true)
    if n == 0:
        return 0.0

    # Weight matrix (quadratic)
    w = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            w[i][j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # Row & column sums
    row_sums = [sum(cm[i]) for i in range(num_classes)]
    col_sums = [sum(cm[i][j] for i in range(num_classes)) for j in range(num_classes)]

    # Expected matrix
    expected = [[row_sums[i] * col_sums[j] / n for j in range(num_classes)] for i in range(num_classes)]

    # Weighted sums
    num = sum(w[i][j] * cm[i][j] for i in range(num_classes) for j in range(num_classes))
    den = sum(w[i][j] * expected[i][j] for i in range(num_classes) for j in range(num_classes))

    return 1.0 - (num / den) if den > 0 else 0.0


def evaluate_classifier(dataset: list[dict]) -> dict:
    """Classifier performansını tüm veri seti üzerinde değerlendir."""
    y_true = []
    y_pred = []
    details = []

    for item in dataset:
        expected = item.get("expected_severity", 5)
        result = classify(item["query"])
        predicted = result.severity

        y_true.append(expected)
        y_pred.append(predicted)
        details.append({
            "id": item["id"],
            "query": item["query"],
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        })

    n = len(y_true)

    # ─── Confusion Matrix ──────────────────────────────────────────────
    cm = [[0] * 5 for _ in range(5)]
    for t, p in zip(y_true, y_pred):
        cm[t - 1][p - 1] += 1

    # ─── Accuracy ──────────────────────────────────────────────────────
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # ─── Per-class Precision, Recall, F1 ───────────────────────────────
    per_class = {}
    for c in range(1, 6):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
            "support": sum(1 for t in y_true if t == c),
        }

    # Macro & Weighted F1
    classes_with_data = [c for c in per_class if per_class[c]["support"] > 0]
    macro_f1 = sum(per_class[c]["f1"] for c in classes_with_data) / max(len(classes_with_data), 1)
    weighted_f1 = sum(
        per_class[c]["f1"] * per_class[c]["support"]
        for c in classes_with_data
    ) / n

    # ─── Undertriage & Overtriage ──────────────────────────────────────
    undertriage = sum(1 for t, p in zip(y_true, y_pred) if t < p)  # Gerçek daha kritik
    overtriage = sum(1 for t, p in zip(y_true, y_pred) if t > p)   # Gerçek daha hafif

    # Kritik undertriage: sev 1-2 olan vakanın 3+ olarak sınıflandırılması
    critical_undertriage = sum(
        1 for t, p in zip(y_true, y_pred) if t <= 2 and p >= 3
    )
    critical_cases = sum(1 for t in y_true if t <= 2)

    # QWK
    qwk = quadratic_weighted_kappa(y_true, y_pred)

    # ─── Rapor ─────────────────────────────────────────────────────────
    severity_labels = {
        1: "KRİTİK", 2: "ACİL", 3: "ÖNCELİKLİ", 4: "GENEL", 5: "BİLGİ"
    }

    print(f"\n{'='*70}")
    print(f"  🚨 Emergency Classifier Evaluation Report")
    print(f"{'='*70}")
    print(f"  Toplam Sorgu: {n}")
    print(f"  Accuracy    : {accuracy:.1%} ({correct}/{n})")
    print(f"  Macro F1    : {macro_f1:.3f}")
    print(f"  Weighted F1 : {weighted_f1:.3f}")
    print(f"  QWK (Kappa) : {qwk:.3f}")
    print(f"{'─'*70}")
    print(f"  Undertriage : {undertriage}/{n} ({undertriage/n:.1%})")
    print(f"  Overtriage  : {overtriage}/{n} ({overtriage/n:.1%})")
    print(f"  ⚠️ Critical Undertriage: {critical_undertriage}/{critical_cases} "
          f"({critical_undertriage/max(critical_cases,1):.1%})")

    print(f"\n{'─'*70}")
    print(f"  📊 Per-Class Metrics")
    print(f"  {'Sev':<4} {'Label':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*46}")
    for c in range(1, 6):
        pc = per_class[c]
        print(f"  {c:<4} {severity_labels[c]:<12} {pc['precision']:>5.1%} {pc['recall']:>5.1%} "
              f"{pc['f1']:>5.3f} {pc['support']:>8}")

    print(f"\n{'─'*70}")
    print(f"  📋 Confusion Matrix")
    print(f"  {'':>14}", end="")
    for c in range(1, 6):
        print(f" P={c:>2}", end="")
    print()
    for r in range(5):
        print(f"  T={r+1} {severity_labels[r+1]:<10}", end="")
        for c in range(5):
            val = cm[r][c]
            marker = " ✓" if r == c and val > 0 else ""
            print(f" {val:>3}{marker}", end="")
        print()

    # Yanlış sınıflandırmalar
    misses = [d for d in details if not d["correct"]]
    if misses:
        print(f"\n{'─'*70}")
        print(f"  ❌ YANLIŞ SINIFLANDIRMALAR ({len(misses)} adet)")
        print(f"  {'─'*66}")
        for m in misses[:15]:
            arrow = "⬆️" if m["expected"] < m["predicted"] else "⬇️"
            print(f"  {arrow} {m['id']:6s} | Beklenen={m['expected']} → Tahmin={m['predicted']} | {m['query'][:50]}")

    print(f"{'='*70}\n")

    report = {
        "total_queries": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "qwk": round(qwk, 4),
        "undertriage_rate": round(undertriage / n, 4),
        "overtriage_rate": round(overtriage / n, 4),
        "critical_undertriage_rate": round(critical_undertriage / max(critical_cases, 1), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "misclassifications": misses,
    }
    return report


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"[ClassifierEval] {len(dataset)} sorgu yüklendi")

    report = evaluate_classifier(dataset)

    export_path = os.path.join(os.path.dirname(__file__), "classifier_results.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ClassifierEval] Sonuçlar kaydedildi: {export_path}")
