from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/Credit_Card_Applications.csv"
OUT = ROOT / "data/processed/credit_card_applications_processed.csv"
FEATURES = [f"A{i}" for i in range(1, 15)]

def main():
    if not RAW.exists(): raise FileNotFoundError(f"Missing dataset: {RAW}")
    df = pd.read_csv(RAW)
    required = FEATURES + ["Class"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Required columns missing: {missing}")
    df = df.drop_duplicates().copy()
    for c in required: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("This project expects a binary Class target containing 0 and 1.")
    # CustomerID is retained in the cleaned file for traceability, but excluded from model features.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} cleaned rows to {OUT}")
    print("Missing feature values are imputed inside each training pipeline to avoid data leakage.")

if __name__ == "__main__": main()
