from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Credit_Card_Applications.csv"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    print("Dataset shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    if "Class" in df:
        print("\nTarget distribution:\n", df["Class"].value_counts(dropna=False).sort_index())
    print("\nFirst five rows:\n", df.head().to_string(index=False))

if __name__ == "__main__": main()
