"""
split_data.py
Splits data/dataset_clean.csv into data/train.csv and data/test.csv.

Method: GROUPED split by hostname. All URLs from one hostname go entirely
into train OR entirely into test, so the model is never tested on a
hostname it has already seen during training.

Before doing that, it measures how many test rows a plain random split
would have leaked, for information only.
"""
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "dataset_clean.csv"
TRAIN_FILE = ROOT / "data" / "train.csv"
TEST_FILE = ROOT / "data" / "test.csv"
SEED = 42
TEST_SIZE = 0.2


def get_host(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def describe(name, part):
    n = len(part)
    phishing = int((part["label"] == 0).sum())
    legit = int((part["label"] == 1).sum())
    print(f"{name}: {n} rows | legitimate (1): {legit} | phishing (0): {phishing}"
          f" | phishing share: {100 * phishing / n:.2f}%")


def main():
    df = pd.read_csv(IN_FILE)
    df["host"] = df["URL"].apply(get_host)

    # Rows with no readable hostname each get their own group
    no_host = (df["host"] == "").to_numpy()
    print("Rows with no readable hostname:", int(no_host.sum()))
    if no_host.any():
        df.loc[no_host, "host"] = ["__nohost__" + str(i) for i in df.index[no_host]]

    print("Total rows:", len(df))
    describe("Whole dataset", df)

    # --- Information only: what a plain random split would leak ---
    r_train, r_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )
    leaked = int(r_test["host"].isin(set(r_train["host"])).sum())
    print("\n[Information only] With a plain random split, test rows whose "
          f"hostname also appears in train: {leaked} of {len(r_test)} "
          f"({100 * leaked / len(r_test):.1f}%)")

    # --- The split we actually use: grouped by hostname ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["host"]))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    # Safety checks: the script stops if any of these fail
    assert set(train["host"]).isdisjoint(set(test["host"])), "Hostname leakage!"
    assert set(train["URL"]).isdisjoint(set(test["URL"])), "URL leakage!"
    assert len(train) + len(test) == len(df), "Rows lost during split!"

    print("\n--- Grouped split (the one we use) ---")
    describe("Train", train)
    describe("Test ", test)
    print(f"Test share of all rows: {100 * len(test) / len(df):.2f}%")
    print("Hostnames in both train and test: 0 (checked)")
    print("URLs in both train and test: 0 (checked)")

    train[["URL", "label"]].to_csv(TRAIN_FILE, index=False)
    test[["URL", "label"]].to_csv(TEST_FILE, index=False)
    print("\nSaved:", TRAIN_FILE)
    print("Saved:", TEST_FILE)


if __name__ == "__main__":
    main()
