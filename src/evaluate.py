from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
FEATURES=[f"A{i}" for i in range(1,15)]
MODEL_FILES={"SVM":"svm_model.pkl","KNN":"knn_model.pkl","Gradient Boosting":"gradient_boosting_model.pkl"}

def main():
    df=pd.read_csv(ROOT/"data/processed/credit_card_applications_processed.csv")
    X_all,y_all=df[FEATURES],df["Class"].astype(int)
    _,X,y_train,y=train_test_split(X_all,y_all,test_size=0.20,random_state=42,stratify=y_all)
    figdir=ROOT/"outputs/figures"; outdir=ROOT/"outputs"
    figdir.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    rows=[]
    for ax,(name,filename) in zip(axes,MODEL_FILES.items()):
        model=joblib.load(ROOT/"models"/filename); pred=model.predict(X); prob=model.predict_proba(X)[:,1]
        cm=confusion_matrix(y,pred,labels=[0,1]); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax)
        ax.set(title=name,xlabel="Predicted class",ylabel="Actual class",xticklabels=["0","1"],yticklabels=["0","1"])
        rows.append({"model":name,"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1_score":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,prob)})
        (outdir/f"{filename.replace('_model.pkl','')}_classification_report.txt").write_text(f"{name} classification report\n\n"+classification_report(y,pred,zero_division=0))
    fig.suptitle("Application Class Prediction – Confusion Matrices"); fig.tight_layout(); fig.savefig(figdir/"approval_prediction_confusion_matrices.png",dpi=160); plt.close(fig)
    metrics=pd.DataFrame(rows).sort_values("f1_score",ascending=False); metrics.to_csv(ROOT/"outputs/model_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,5)); metrics.set_index("model")[["accuracy","precision","recall","f1_score","roc_auc"]].plot(kind="bar",ax=ax)
    ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("SVM vs KNN vs Gradient Boosting – Model Performance"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(figdir/"algorithm_performance_comparison.png",dpi=160); plt.close(fig)
    # Model-native feature importance is available for Gradient Boosting.
    gb=joblib.load(ROOT/"models/gradient_boosting_model.pkl")
    importances=gb.named_steps["classifier"].feature_importances_
    imp=pd.Series(importances,index=FEATURES).sort_values()
    fig,ax=plt.subplots(figsize=(8,6)); imp.plot(kind="barh",ax=ax); ax.set(title="Gradient Boosting – Applicant Feature Importance",xlabel="Importance",ylabel="Dataset feature"); fig.tight_layout(); fig.savefig(figdir/"gradient_boosting_feature_importance.png",dpi=160); plt.close(fig)
    best=metrics.iloc[0]
    print("Evaluation complete. Figures:",figdir)

if __name__=="__main__": main()
