"""
train_model.py
Trains a Random Forest on data/train.csv, evaluates it on data/test.csv,
saves the model and writes the REAL results into README.md.

Run:  python src/train_model.py
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from feature_extraction import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "train.csv"
TEST_FILE = ROOT / "data" / "test.csv"
MODEL_FILE = ROOT / "model" / "phishing_model.joblib"
FEATURE_FILE = ROOT / "model" / "feature_names.txt"
METRICS_FILE = ROOT / "model" / "metrics.json"
README_FILE = ROOT / "README.md"

PHISHING = 0      # dataset label meaning: 0 = phishing, 1 = legitimate
LEGITIMATE = 1

# Small settings on purpose, so the model file stays small
MODEL_SETTINGS = dict(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


def load_xy(path):
    df = pd.read_csv(path, dtype={"URL": str})
    df = df.dropna(subset=["URL", "label"])
    print(f"Extracting features from {path.name} ({len(df)} URLs)...")
    start = time.time()
    X = np.array([extract_features(u) for u in df["URL"]], dtype=float)
    y = df["label"].astype(int).to_numpy()
    print(f"  done in {time.time() - start:.1f} seconds")
    return X, y


def update_readme(results_markdown):
    """Replace the text between the two RESULTS markers in README.md."""
    if not README_FILE.exists():
        print("README.md not found - skipping README update.")
        return
    text = README_FILE.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- RESULTS_START -->)(.*?)(<!-- RESULTS_END -->)", re.S)
    if not pattern.search(text):
        print("Markers not found in README.md - skipping README update.")
        return
    new_text = pattern.sub(lambda m: m.group(1) + "\n" + results_markdown + "\n" + m.group(3), text)
    README_FILE.write_text(new_text, encoding="utf-8")
    print("README.md results section updated with the real numbers.")


def main():
    X_train, y_train = load_xy(TRAIN_FILE)
    X_test, y_test = load_xy(TEST_FILE)

    print("\nTraining Random Forest...", MODEL_SETTINGS)
    model = RandomForestClassifier(**MODEL_SETTINGS)
    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start
    print(f"Training finished in {train_seconds:.1f} seconds")

    # ---- Evaluation on the unseen test set (phishing = positive class) ----
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, pos_label=PHISHING, zero_division=0)
    rec = recall_score(y_test, pred, pos_label=PHISHING, zero_division=0)
    f1 = f1_score(y_test, pred, pos_label=PHISHING, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[PHISHING, LEGITIMATE])
    train_acc = accuracy_score(y_train, model.predict(X_train))
    majority_share = max(np.mean(y_test == PHISHING), np.mean(y_test == LEGITIMATE))

    phish_ok, phish_missed = int(cm[0][0]), int(cm[0][1])
    legit_flagged, legit_ok = int(cm[1][0]), int(cm[1][1])

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                         key=lambda p: p[1], reverse=True)

    # ---- Save model + feature names ----
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    FEATURE_FILE.write_text("\n".join(FEATURE_NAMES) + "\n")
    size_mb = MODEL_FILE.stat().st_size / 1024 / 1024

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "model_settings": MODEL_SETTINGS,
        "test_accuracy": acc,
        "phishing_precision": prec,
        "phishing_recall": rec,
        "phishing_f1": f1,
        "train_accuracy": train_acc,
        "majority_class_baseline_accuracy": float(majority_share),
        "confusion_matrix_labels_[phishing,legitimate]": cm.tolist(),
        "model_file_size_mb": round(size_mb, 2),
        "feature_importances": {n: float(v) for n, v in importances},
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    # ---- Print results ----
    print("\n========== RESULTS ON THE TEST SET ==========")
    print(f"Accuracy:                   {acc:.4f}")
    print(f"Precision (phishing class): {prec:.4f}")
    print(f"Recall    (phishing class): {rec:.4f}")
    print(f"F1-score  (phishing class): {f1:.4f}")
    print(f"Training-set accuracy:      {train_acc:.4f}")
    print(f"Always-guess-majority baseline accuracy: {majority_share:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print("                     Pred Phishing   Pred Legitimate")
    print(f"Actual Phishing      {phish_ok:13d}   {phish_missed:15d}")
    print(f"Actual Legitimate    {legit_flagged:13d}   {legit_ok:15d}")
    print("\nTop 10 most important features:")
    for name, value in importances[:10]:
        print(f"  {name:26s} {value:.4f}")
    print(f"\nModel saved: {MODEL_FILE}  ({size_mb:.2f} MB)")
    print(f"Features saved: {FEATURE_FILE}")
    print(f"Metrics saved: {METRICS_FILE}")
    if size_mb > 40:
        print("WARNING: model is larger than 40 MB. Tell your helper so settings can be reduced.")

    # ---- README section ----
    lines = [
        f"*Generated automatically by `src/train_model.py` on {metrics['trained_at']}. "
        "These are the real numbers from the test set.*",
        "",
        f"- Training rows: **{len(y_train)}**, test rows: **{len(y_test)}**",
        "- Positive class for precision / recall / F1: **Phishing**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {acc:.4f} |",
        f"| Precision (phishing) | {prec:.4f} |",
        f"| Recall (phishing) | {rec:.4f} |",
        f"| F1-score (phishing) | {f1:.4f} |",
        f"| Training-set accuracy | {train_acc:.4f} |",
        f"| Baseline: always predict the bigger class | {majority_share:.4f} |",
        "",
        "**Confusion matrix** (rows = actual, columns = predicted):",
        "",
        "| | Predicted Phishing | Predicted Legitimate |",
        "|---|---|---|",
        f"| **Actual Phishing** | {phish_ok} | {phish_missed} |",
        f"| **Actual Legitimate** | {legit_flagged} | {legit_ok} |",
        "",
        f"- Phishing URLs correctly caught: {phish_ok}; phishing URLs missed: {phish_missed}",
        f"- Legitimate URLs wrongly flagged as phishing: {legit_flagged}; correctly accepted: {legit_ok}",
        "",
        "**Top 10 most important features (from the trained model):**",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for name, value in importances[:10]:
        lines.append(f"| {name} | {value:.4f} |")
    lines += [
        "",
        f"Model settings: `{MODEL_SETTINGS}`  ",
        f"Saved model size: {size_mb:.2f} MB",
    ]
    update_readme("\n".join(lines))


if __name__ == "__main__":
    main()
