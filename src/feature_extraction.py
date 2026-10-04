"""
feature_extraction.py
Turns a URL string into a list of numbers (features) for the ML model.

Only the TEXT of the URL is used. Nothing is downloaded and no website
is opened. The same function is used for training and for prediction,
so the feature order is always identical.
"""
import ipaddress
import math
from urllib.parse import urlparse

# Words that often appear in phishing URLs (a simple, explainable list)
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "update", "secure", "account", "bank",
    "password", "confirm", "paypal", "webscr", "ebay", "suspend",
    "wallet", "billing",
]

# The ORDER of this list is the ORDER of the numbers returned below.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at_symbols",
    "num_percent",
    "num_digits",
    "num_letters",
    "hostname_digits",
    "hostname_hyphens",
    "is_https",
    "is_http",
    "has_ip_address",
    "has_port",
    "subdomain_count",
    "url_entropy",
    "suspicious_keyword_count",
]


def _shannon_entropy(text):
    """How 'random' the characters look. Higher = more random-looking."""
    if not text:
        return 0.0
    total = len(text)
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _parse(url):
    """Parse safely. Badly formed URLs give empty parts instead of crashing."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        scheme = (parsed.scheme or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        try:
            has_port = 1 if parsed.port is not None else 0
        except ValueError:
            has_port = 1  # a port was written but it is not a valid number
        return scheme, host, path, query, has_port
    except ValueError:
        return "", "", "", "", 0


def extract_features(url):
    """Return a list of numbers, in the same order as FEATURE_NAMES."""
    url = str(url).strip()
    scheme, host, path, query, has_port = _parse(url)
    if path == "/" and not query and url.endswith("/"):
        url = url[:-1]
        scheme, host, path, query, has_port = _parse(url)
    lower_url = url.lower()

    has_ip = 1 if _is_ip(host) else 0

    # Subdomains: number of dot-separated parts in the hostname minus 2
    # (the main name and the ending such as .com). IP addresses have none.
    if host and not has_ip:
        subdomains = max(len(host.split(".")) - 2, 0)
    else:
        subdomains = 0

    keyword_count = sum(lower_url.count(word) for word in SUSPICIOUS_KEYWORDS)

    features = [
        len(url),
        len(host),
        len(path),
        len(query),
        url.count("."),
        url.count("-"),
        url.count("_"),
        url.count("/"),
        url.count("@"),
        url.count("%"),
        sum(ch.isdigit() for ch in url),
        sum(ch.isalpha() for ch in url),
        sum(ch.isdigit() for ch in host),
        host.count("-"),
        1 if scheme == "https" else 0,
        1 if scheme == "http" else 0,
        has_ip,
        has_port,
        subdomains,
        _shannon_entropy(url),
        keyword_count,
    ]
    assert len(features) == len(FEATURE_NAMES)
    return [float(x) for x in features]


if __name__ == "__main__":
    demo = "https://www.example.com/login?user=1"
    for name, value in zip(FEATURE_NAMES, extract_features(demo)):
        print(f"{name:26s} {value}")
