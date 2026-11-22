from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import sys
import sklearn._loss._loss as _loss_module
sys.modules["_loss"] = _loss_module
ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/"models"
INFO_PATH=MODEL_DIR/"best_model_info.json"
st.set_page_config(page_title="Credit Card Application Prediction",page_icon="💳",layout="wide")
st.title("Credit Card Application Prediction")
st.caption("Machine-learning classification using SVM, KNN and Gradient Boosting")

@st.cache_resource
def load_assets():
    if not INFO_PATH.exists() or not (MODEL_DIR/"best_model.pkl").exists():
        return None,None
    info=json.loads(INFO_PATH.read_text())
    return joblib.load(MODEL_DIR/"best_model.pkl"),info
model,info=load_assets()
if model is None:
    st.error("Model files are missing. Run the training commands in README.md first.")
    st.stop()

with st.sidebar:
    st.header("Project information")
    st.write("**Selected model:**",info["best_model"].replace("_"," ").title())
    st.write("**Input features:** A1–A14")
    st.write("**Target:** Class (0 or 1)")
    st.info("Class 0 and Class 1 are shown as dataset labels. Their real-world approval meaning must be confirmed from the dataset documentation.")

tab1,tab2,tab3=st.tabs(["Single prediction","Batch CSV prediction","Model results"])
features=info["features"]
with tab1:
    st.subheader("Enter applicant feature values")
    st.write("Enter values for A1 through A14. The fields start at 0; adjust them to match the feature values in your dataset.")
    with st.form("single_prediction"):
        cols=st.columns(2)
        values={}
        for i,f in enumerate(features):
            with cols[i%2]: values[f]=st.number_input(f, value=0.0, format="%.4f", key=f"single_{f}")
        submitted=st.form_submit_button("Predict class",type="primary")
    if submitted:
        row=pd.DataFrame([[values[f] for f in features]],columns=features)
        prediction=int(model.predict(row)[0])
        probability=float(model.predict_proba(row)[0][1]) if hasattr(model,"predict_proba") else None
        st.success(f"Predicted dataset label: Class {prediction}")
        if probability is not None: st.metric("Estimated probability of Class 1",f"{probability:.1%}")
        st.caption("This is a model classification, not a real credit decision.")
with tab2:
    st.subheader("Upload multiple applications")
    st.write("CSV must contain columns A1 through A14. CustomerID and Class are optional and ignored for prediction.")
    sample=pd.DataFrame([{f:0 for f in features}])
    st.download_button("Download sample input CSV",sample.to_csv(index=False).encode("utf-8"),"prediction_input_template.csv","text/csv")
    uploaded=st.file_uploader("Choose a CSV file",type=["csv"])
    if uploaded is not None:
        try:
            incoming=pd.read_csv(uploaded)
            missing=[f for f in features if f not in incoming.columns]
            if missing: st.error("Missing required columns: "+", ".join(missing))
            else:
                X=incoming[features].apply(pd.to_numeric,errors="coerce")
                preds=model.predict(X)
                result=incoming.copy(); result["Predicted_Class"]=preds
                if hasattr(model,"predict_proba"): result["Probability_Class_1"]=model.predict_proba(X)[:,1]
                st.dataframe(result.head(100),use_container_width=True)
                st.download_button("Download predictions",result.to_csv(index=False).encode("utf-8"),"application_predictions.csv","text/csv")
        except Exception as e: st.error(f"Could not process CSV: {e}")
with tab3:
    st.subheader("Model comparison")
    metrics_path=ROOT/"outputs/model_comparison.csv"
    if metrics_path.exists():
        metrics=pd.read_csv(metrics_path); st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.select_dtypes('number').columns}),use_container_width=True)
    else: st.warning("Run src/train.py and src/evaluate.py to create evaluation results.")
    figdir=ROOT/"outputs/figures"
    for filename,title in [("application_approval_distribution.png","Dataset class distribution"),("applicant_feature_correlation.png","Feature correlation"),("approval_prediction_confusion_matrices.png","Confusion matrices"),("algorithm_performance_comparison.png","Algorithm comparison"),("gradient_boosting_feature_importance.png","Gradient Boosting feature importance")]:
        path=figdir/filename
        if path.exists(): st.image(str(path),caption=title,use_container_width=True)
