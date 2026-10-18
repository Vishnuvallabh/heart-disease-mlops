"""Load the raw UCI Cleveland file, clean it, and save heart_clean.csv."""
import os

import pandas as pd

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

RAW_PATH = "data/raw/processed.cleveland.data"
CLEAN_PATH = "data/processed/heart_clean.csv"


def load_raw(path=RAW_PATH):
    """Read the raw file: no header row, '?' means missing."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df):
    """Fill missing ca/thal with the mode and turn num (0-4) into a 0/1 target."""
    df = df.copy()
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


def main():
    df = clean(load_raw())
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {len(df)} rows to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
