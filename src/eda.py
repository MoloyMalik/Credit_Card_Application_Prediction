from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/credit_card_applications_processed.csv"
FIG = ROOT / "outputs/figures"
FEATURES = [f"A{i}" for i in range(1, 15)]

def main():
    if not DATA.exists():
        from prepare_data import main as prepare
        prepare()
    df = pd.read_csv(DATA)
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    counts = df["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"Class {i}" for i in counts.index], counts.values)
    ax.set(title="Credit Card Application Class Distribution", xlabel="Dataset class", ylabel="Number of applications")
    for p, v in zip(ax.patches, counts.values): ax.annotate(str(v), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(FIG/"application_approval_distribution.png", dpi=160); plt.close(fig)

    corr = df[FEATURES].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Applicant Feature Correlation (A1–A14)")
    fig.tight_layout(); fig.savefig(FIG/"applicant_feature_correlation.png", dpi=160); plt.close(fig)
    print("EDA figures saved in", FIG)

if __name__ == "__main__": main()
