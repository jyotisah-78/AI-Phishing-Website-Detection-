import sys
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading PhiUSIIL dataset from UCI...")
    ds = fetch_ucirepo(id=967)

    parts = [ds.data.features, ds.data.targets]

    if ds.data.ids is not None:
        parts.append(ds.data.ids)

    df = pd.concat([p for p in parts if p is not None], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    cols = {c.lower(): c for c in df.columns}

    if "url" not in cols or "label" not in cols:
        print("ERROR: URL or label column not found.")
        print("Columns found:", list(df.columns))
        sys.exit(1)

    out = df[[cols["url"], cols["label"]]].copy()
    out.columns = ["URL", "label"]

    out = out.dropna()
    out["label"] = out["label"].astype(int)

    out.to_csv(OUT_FILE, index=False)

    print("\nDataset downloaded successfully!")
    print("Saved to:", OUT_FILE)
    print("File size (MB):", round(OUT_FILE.stat().st_size / 1024 / 1024, 2))
    print("Total rows:", len(out))
    print("\nRows per label:")
    print(out["label"].value_counts().to_string())
    print("\nDuplicate URLs:", int(out["URL"].duplicated().sum()))

    for lab in sorted(out["label"].unique()):
        print(f"\n--- Sample URLs with label {lab} ---")
        samples = out[out["label"] == lab]["URL"].sample(
            min(5, len(out[out["label"] == lab])),
            random_state=1
        )

        for url in samples:
            print(url)


if __name__ == "__main__":
    main()
