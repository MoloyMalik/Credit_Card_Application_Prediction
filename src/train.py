from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/credit_card_applications_processed.csv"
MODELS = ROOT / "models"
FEATURES = [f"A{i}" for i in range(1, 15)]
RANDOM_STATE = 42

def main():
    if not DATA.exists():
        from prepare_data import main as prepare
        prepare()
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    MODELS.mkdir(parents=True, exist_ok=True)
    definitions = {
        "svm": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
        "knn": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
    }
    rows=[]
    for name, model in definitions.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS/f"{name}_model.pkl")
        pred=model.predict(X_test)
        prob=model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        rows.append({"model": name.replace("_", " ").title(), "accuracy": accuracy_score(y_test,pred), "precision": precision_score(y_test,pred,zero_division=0), "recall": recall_score(y_test,pred,zero_division=0), "f1_score": f1_score(y_test,pred,zero_division=0), "roc_auc": roc_auc_score(y_test,prob) if prob is not None else float("nan")})
    metrics=pd.DataFrame(rows).sort_values(["f1_score","roc_auc","accuracy"], ascending=False)
    best_name=metrics.iloc[0]["model"].lower().replace(" ","_")
    joblib.dump(definitions[best_name], MODELS/"best_model.pkl")
    info={"best_model": best_name, "selection_metric": "F1-score (tie-breakers: ROC-AUC, accuracy)", "features": FEATURES, "target": "Class", "class_labels": {"0":"Class 0", "1":"Class 1"}, "test_size":0.2, "random_state":RANDOM_STATE, "dataset_rows":int(len(df))}
    (MODELS/"best_model_info.json").write_text(json.dumps(info, indent=2))
    (ROOT/"outputs").mkdir(exist_ok=True)
    metrics.to_csv(ROOT/"outputs/model_comparison.csv", index=False)
    print(metrics.to_string(index=False))
    print("Best model:", best_name)

if __name__ == "__main__": main()
