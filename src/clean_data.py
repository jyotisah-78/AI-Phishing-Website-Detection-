from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "dataset.csv"
OUT_FILE = ROOT / "data" / "dataset_clean.csv"

def get_host(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""

def main():
    df = pd.read_csv(IN_FILE)
    n_start = len(df)
    print("Rows at start:", n_start)

    df["URL"] = df["URL"].astype(str).str.strip()
    empty = int((df["URL"] == "").sum())
    df = df[df["URL"] != ""]
    print("Empty URLs removed:", empty)

    labels_per_url = df.groupby("URL")["label"].nunique()
    conflict_urls = labels_per_url[labels_per_url > 1].index
    n_conflict_rows = int(df["URL"].isin(conflict_urls).sum())
    df = df[~df["URL"].isin(conflict_urls)]
    print("URLs with conflicting labels:", len(conflict_urls),
          "(rows removed:", n_conflict_rows, ")")

    n_dup = int(df["URL"].duplicated().sum())
    df = df.drop_duplicates(subset="URL", keep="first")
    print("Exact duplicate rows removed:", n_dup)

    assert df["URL"].duplicated().sum() == 0

    df = df.reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)

    print("\nRows at end:", len(df))
    print("Total rows removed:", n_start - len(df))
    print("Rows per label after cleaning:")
    print(df["label"].value_counts().to_string())
    print("Saved to:", OUT_FILE)

    df["host"] = df["URL"].apply(get_host)
    host_counts = df["host"].value_counts()
    n_hosts = len(host_counts)
    shared_hosts = host_counts[host_counts > 1]
    rows_in_shared = int(shared_hosts.sum())
    labels_per_host = df.groupby("host")["label"].nunique()
    mixed_hosts = int((labels_per_host > 1).sum())

    print("\n--- Domain overlap check ---")
    print("Unique hostnames:", n_hosts)
    print("Hostnames appearing in more than one row:", len(shared_hosts))
    print("Rows belonging to those shared hostnames:", rows_in_shared,
          f"({100 * rows_in_shared / len(df):.1f}% of all rows)")
    print("Hostnames that have BOTH labels:", mixed_hosts)
    print("Most common hostnames:")
    print(host_counts.head(5).to_string())

if __name__ == "__main__":
    main()
