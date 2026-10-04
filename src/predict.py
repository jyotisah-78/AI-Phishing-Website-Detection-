"""
predict.py
Loads the saved model and predicts Phishing / Legitimate for one URL.

The prediction uses ONLY the text of the URL. The website is never opened
and no online security database is checked.

Command line use:
    python src/predict.py "https://www.example.com/login"
"""
import sys
from pathlib import Path

import joblib
import numpy as np
# The next import looks unused, but it makes sure PyInstaller packs the
# Random Forest code into the EXE so the saved model can be loaded there.
from sklearn.ensemble import RandomForestClassifier  # noqa: F401

from feature_extraction import FEATURE_NAMES, extract_features

PHISHING = 0      # label meaning in the dataset
LEGITIMATE = 1


def get_base_dir():
    """Project folder normally; PyInstaller's temp folder inside the EXE."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


MODEL_PATH = get_base_dir() / "model" / "phishing_model.joblib"
FEATURE_FILE = get_base_dir() / "model" / "feature_names.txt"


def validate_url(url):
    """Return the cleaned URL or raise ValueError with a clear message."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please enter a URL.")
    url = url.strip()
    if " " in url:
        raise ValueError("A URL cannot contain spaces.")
    lower = url.lower()
    if not (lower.startswith("http://") or lower.startswith("https://")):
        raise ValueError("Please enter the full URL starting with http:// or https://")
    return url


class PhishingDetector:
    def __init__(self, model_path=MODEL_PATH, feature_file=FEATURE_FILE):
        model_path = Path(model_path)
        feature_file = Path(feature_file)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Train it first with:  python src/train_model.py"
            )
        self.model = joblib.load(model_path)

        # Safety check: the model must have been trained with the same features
        if feature_file.exists():
            saved = [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]
            if saved != FEATURE_NAMES:
                raise RuntimeError(
                    "model/feature_names.txt does not match feature_extraction.py. "
                    "Please retrain the model: python src/train_model.py"
                )
        self.phishing_column = list(self.model.classes_).index(PHISHING)

    def predict(self, url):
        url = validate_url(url)
        x = np.array([extract_features(url)])
        proba = self.model.predict_proba(x)[0]
        phishing_probability = float(proba[self.phishing_column])
        is_phishing = phishing_probability >= 0.5
        confidence = phishing_probability if is_phishing else 1.0 - phishing_probability
        return {
            "url": url,
            "label": "Phishing" if is_phishing else "Legitimate",
            "is_phishing": bool(is_phishing),
            "phishing_probability": phishing_probability,
            "confidence": float(confidence),
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "https://www.example.com/login"')
        sys.exit(1)
    try:
        result = PhishingDetector().predict(sys.argv[1])
    except (ValueError, FileNotFoundError, RuntimeError) as err:
        print("Error:", err)
        sys.exit(1)
    print("URL:                  ", result["url"])
    print("Prediction:           ", result["label"])
    print(f"Confidence:            {result['confidence'] * 100:.1f}%")
    print(f"Phishing probability:  {result['phishing_probability'] * 100:.1f}%")
    print("Note: ML prediction from URL text only. Not a guarantee.")
