"""
test_project.py
Simple automated tests. Run with:  python src/test_project.py

Prediction tests check that the output is VALID (correct keys, probabilities
between 0 and 1). They do NOT hard-code which label a URL must get.
The model must be trained first:  python src/train_model.py
"""
import math
import unittest

from feature_extraction import FEATURE_NAMES, extract_features
from predict import FEATURE_FILE, MODEL_PATH, PhishingDetector

NORMAL_URL = "https://www.wikipedia.org/"
SUSPICIOUS_URL = "http://secure-login.verify-account.example-bank.xyz/webscr?cmd=login&update=1"
MODEL_MISSING_MSG = "Model file not found. Run first:  python src/train_model.py"


def feature_dict(url):
    return dict(zip(FEATURE_NAMES, extract_features(url)))


class TestFeatureExtraction(unittest.TestCase):
    def test_feature_count_and_numbers(self):
        values = extract_features(NORMAL_URL)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        for v in values:
            self.assertTrue(math.isfinite(v))

    def test_known_values(self):
        f = feature_dict("https://www.example.com/login?x=1")
        self.assertEqual(f["is_https"], 1)
        self.assertEqual(f["is_http"], 0)
        self.assertEqual(f["hostname_length"], len("www.example.com"))
        self.assertEqual(f["path_length"], len("/login"))
        self.assertEqual(f["query_length"], len("x=1"))
        self.assertEqual(f["num_dots"], 2)
        self.assertEqual(f["subdomain_count"], 1)
        self.assertEqual(f["suspicious_keyword_count"], 1)
        self.assertGreater(f["url_entropy"], 0)

    def test_root_trailing_slash_does_not_change_features(self):
        without_slash = extract_features("https://www.wikipedia.org")
        with_slash = extract_features("https://www.wikipedia.org/")
        self.assertEqual(with_slash, without_slash)

    def test_ip_address_and_port(self):
        f = feature_dict("http://192.168.1.1:8080/login")
        self.assertEqual(f["has_ip_address"], 1)
        self.assertEqual(f["has_port"], 1)
        self.assertEqual(f["is_http"], 1)
        self.assertEqual(f["subdomain_count"], 0)

    def test_special_characters(self):
        f = feature_dict("http://user@my-site.com/a_b%20c")
        self.assertEqual(f["num_at_symbols"], 1)
        self.assertEqual(f["num_percent"], 1)
        self.assertEqual(f["num_underscores"], 1)
        self.assertEqual(f["hostname_hyphens"], 1)

    def test_badly_formed_url_does_not_crash(self):
        values = extract_features("http://[bad-url")
        self.assertEqual(len(values), len(FEATURE_NAMES))


class TestModelAndPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert MODEL_PATH.exists(), MODEL_MISSING_MSG
        cls.detector = PhishingDetector()

    def test_model_loads_and_feature_order_matches(self):
        saved = [l.strip() for l in FEATURE_FILE.read_text().splitlines() if l.strip()]
        self.assertEqual(saved, FEATURE_NAMES)
        self.assertEqual(self.detector.model.n_features_in_, len(FEATURE_NAMES))

    def _check_result(self, url):
        r = self.detector.predict(url)
        print(f"\n  {url}\n  -> {r['label']} (confidence {r['confidence']:.3f})")
        self.assertIn(r["label"], ("Phishing", "Legitimate"))
        self.assertEqual(r["is_phishing"], r["label"] == "Phishing")
        self.assertGreaterEqual(r["phishing_probability"], 0.0)
        self.assertLessEqual(r["phishing_probability"], 1.0)
        self.assertGreaterEqual(r["confidence"], 0.5)
        self.assertLessEqual(r["confidence"], 1.0)

    def test_normal_looking_url(self):
        self._check_result(NORMAL_URL)

    def test_suspicious_looking_url(self):
        self._check_result(SUSPICIOUS_URL)

    def test_empty_input(self):
        for bad in ["", "   ", None]:
            with self.assertRaises(ValueError):
                self.detector.predict(bad)

    def test_url_without_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.predict("example.com")


if __name__ == "__main__":
    unittest.main(verbosity=2)
