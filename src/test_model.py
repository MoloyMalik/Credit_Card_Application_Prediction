from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]

def main():
    info=json.loads((ROOT/"models/best_model_info.json").read_text())
    model=joblib.load(ROOT/"models/best_model.pkl")
    df=pd.read_csv(ROOT/"data/processed/credit_card_applications_processed.csv")
    X,_,y,_=train_test_split(df[info["features"]],df["Class"].astype(int),test_size=0.20,random_state=42,stratify=df["Class"].astype(int))
    sample=X.head(5).copy(); sample["Class"]=y.head(5).values
    pred=model.predict(sample[info["features"]])
    print("Best model:",info["best_model"])
    print(pd.DataFrame({"actual_class":sample["Class"],"predicted_class":pred}).to_string(index=False))

if __name__=="__main__": main()
