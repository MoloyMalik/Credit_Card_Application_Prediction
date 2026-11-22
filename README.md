# 💳 Credit Card Application Prediction Using Machine Learning

## 📌 1. Project Overview

The **Credit Card Application Prediction** is a Machine Learning-based project designed to classify credit card application records using applicant-related information.

The system analyzes application attributes and predicts the target class using three different Machine Learning algorithms:

- 🤖 **Support Vector Machine (SVM)**
- 🤖 **K-Nearest Neighbors (KNN)**
- 🤖 **Gradient Boosting Classifier**

The main purpose is to analyze historical credit card application data, identify patterns, compare different classification algorithms, and select the best-performing model.

The selected model is integrated into a Streamlit web application to provide an interactive prediction interface.

---

## 🎯 2. Problem Statement

Credit card application datasets contain different attributes related to applicants. Analyzing these attributes can help identify patterns associated with different application classes.

Developing an effective classification model requires proper data preprocessing, feature analysis, algorithm selection, and model evaluation.

To address this problem, this project develops a Machine Learning-based prediction system that analyzes applicant information and predicts the target class of credit card application records.

### 🔍 Problems Addressed

- Analyze credit card application records.
- Identify patterns in applicant-related attributes.
- Perform data preprocessing and feature analysis.
- Compare different Machine Learning algorithms.
- Evaluate classification model performance.
- Develop an interactive prediction interface.

The system provides predictions based on patterns learned from the available dataset. It is an educational project and does not replace actual financial approval procedures.

---

## 🎯 3. Project Objectives

- Inspect and understand the credit card application dataset.
- Perform data preprocessing and cleaning.
- Analyze applicant information using Exploratory Data Analysis (EDA).
- Implement and compare three Machine Learning algorithms.
- Evaluate model performance using standard classification metrics.
- Select and save the best-performing model.
- Generate graphs for data analysis and model comparison.
- Develop a user-friendly Streamlit web application.

---

## 📊 4. Dataset Description

**Dataset:** Credit Card Applications Dataset

The dataset contains application records with different applicant-related attributes.

### Dataset Information

- 📌 Total Records: 690
- 📌 Total Columns: 16
- 📌 Input Features: 14
- 📌 Identifier Column: CustomerID
- 📌 Target Variable: Class

The dataset contains applicant-related attributes represented by columns such as A1, A2, A3, and so on.

### 🎯 Target Classification

The `Class` column is used as the target variable for Machine Learning classification.

| Attribute | Description |
|:---|:---|
| CustomerID | Unique application identifier |
| A1 to A14 | Input features |
| Class | Target classification variable |

The CustomerID column is excluded from model training because it is an identifier rather than a predictive feature.

The original target class labels are retained for prediction. They are not automatically interpreted as approved or rejected applications.

---

## 🛠️ 5. Technologies Used

| Technology | Purpose |
|:---|:---|
| 🐍 Python | Programming Language |
| 🐼 Pandas | Data Processing and Analysis |
| 🔢 NumPy | Numerical Operations |
| 📈 Matplotlib | Data Visualization |
| 📊 Seaborn | Statistical Visualization |
| 🤖 Scikit-learn | Machine Learning |
| 🌐 Streamlit | Web Application |
| 💾 Joblib | Model Saving and Loading |

---

## 🔄 6. Project Workflow

The project follows these main steps:

1. 🔍 **Data Inspection:** Understanding the dataset, checking data types, missing values, and data structure.
2. 🧹 **Data Preprocessing:** Cleaning the dataset, handling missing values, and preparing input features.
3. 📊 **Exploratory Data Analysis:** Analyzing application records and feature relationships through different graphs.
4. ✂️ **Feature and Target Separation:** Separating input features (A1–A14) and target variable (Class).
5. 📂 **Train-Test Splitting:** Dividing the dataset into training and testing subsets using stratified sampling.
6. 🤖 **Model Training:** Training three Machine Learning classification algorithms.
7. 📈 **Model Evaluation:** Comparing model performance using standard evaluation metrics.
8. 🏆 **Best Model Selection:** Selecting the best-performing model based on evaluation results.
9. 💾 **Model Saving:** Saving trained models and model information.
10. 🧪 **Model Testing:** Loading the saved model and checking its prediction functionality.
11. 🌐 **Web Application:** Developing an interactive Streamlit application for credit card application prediction.

---

## 🤖 7. Machine Learning Algorithms

Three classification algorithms are implemented in this project:

1. Support Vector Machine (SVM)
2. K-Nearest Neighbors (KNN)
3. Gradient Boosting Classifier

### ⚙️ Model Configuration

The models use an 80:20 train-test split with stratified sampling.

The preprocessing pipelines use:

- Median imputation for handling missing values.
- StandardScaler for SVM and KNN.
- Median imputation for Gradient Boosting.

### 🔍 Algorithm Explanation

**1. Support Vector Machine (SVM)**

SVM is a supervised Machine Learning algorithm that identifies an optimal decision boundary between different classes.

**2. K-Nearest Neighbors (KNN)**

KNN classifies a new application record based on the classes of its nearest training data points.

**3. Gradient Boosting Classifier**

Gradient Boosting is an ensemble learning algorithm that combines multiple decision trees sequentially to improve prediction performance.

---

## 📈 8. Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix

### 🏆 Model Performance Comparison

The performance of all three algorithms is compared using the same testing dataset.

| Model | Evaluation |
|:---|:---|
| SVM | Evaluated using classification metrics |
| KNN | Evaluated using classification metrics |
| Gradient Boosting | Evaluated using classification metrics |

The detailed numerical results are saved in `model_comparison.csv`.

### ⭐ Selected Model

The best-performing model is selected based on the highest F1-score, with ROC-AUC and accuracy used as tie-breakers.

The selected model is saved as:

`models/best_model.pkl`

The selected model information is stored in:

`models/best_model_info.json`

---

## 📊 9. Output Graphs

The project generates different graphs to understand the dataset and evaluate model performance.

The generated graphs include:

- 📊 **application_approval_distribution.png:** Shows the distribution of target classes in the dataset.
- 📈 **applicant_feature_correlation.png:** Shows the correlation between numerical applicant features.
- 📉 **approval_prediction_confusion_matrices.png:** Shows the classification results of the three Machine Learning models.
- 📊 **algorithm_performance_comparison.png:** Compares the performance of SVM, KNN, and Gradient Boosting.
- 🎯 **gradient_boosting_feature_importance.png:** Shows the relative importance of input features in the Gradient Boosting model.

The model comparison results are also saved in `model_comparison.csv`.

All generated graphs are stored inside the `outputs/figures/` directory.

---

## 🌐 10. Web Application

A user-friendly web application is developed using Streamlit.

### ✨ Key Features

- 📝 Applicant information input interface.
- 🤖 Machine Learning-based application classification.
- 💾 Integration with the saved best-performing model.
- 📊 Display of predicted target class.
- 💻 Simple and interactive user interface.
- ⚡ Prediction without retraining the model during normal application use.

The application allows users to enter applicant information and receive a predicted target class using the trained Machine Learning model.

---

## 📁 11. Project Structure

```text
Credit_Card_Application_Prediction/
│
├── data/
│   ├── raw/
│   │   └── Credit_Card_Applications.csv
│   │
│   └── processed/
│       └── credit_card_applications_processed.csv
│
├── models/
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── best_model.pkl
│   └── best_model_info.json
│
├── outputs/
│   ├── figures/
│   │   ├── application_approval_distribution.png
│   │   ├── applicant_feature_correlation.png
│   │   ├── approval_prediction_confusion_matrices.png
│   │   ├── algorithm_performance_comparison.png
│   │   └── gradient_boosting_feature_importance.png
│   │
│   ├── model_comparison.csv
│   ├── svm_classification_report.txt
│   ├── knn_classification_report.txt
│   └── gradient_boosting_classification_report.txt
│
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── test_model.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ 12. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/MoloyMalik/Credit_Card_Application_Prediction.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Credit_Card_Application_Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run Data Inspection

```bash
python src/inspect_data.py
```

### Step 7: Run Data Preprocessing

```bash
python src/prepare_data.py
```

### Step 8: Run Exploratory Data Analysis

```bash
python src/eda.py
```

### Step 9: Train the Models

```bash
python src/train.py
```

### Step 10: Evaluate the Models

```bash
python src/evaluate.py
```

### Step 11: Test the Saved Model

```bash
python src/test_model.py
```

### Step 12: Start the Web Application

```bash
streamlit run app.py
```

The Streamlit application will provide a local URL. Open the URL in your browser to access the prediction interface.

Make sure the dataset is available in the correct folder before running the project.

---

## ⚠️ 13. Project Limitations

- The dataset contains only 690 application records.
- The dataset may not represent every real-world credit card applicant.
- The model uses only the available input features.
- Model performance depends on data quality and preprocessing.
- Predictions may vary when new application records are introduced.
- The target classification should be interpreted according to the original dataset documentation.
- Predictions are not guaranteed to be correct for every application.
- The project is intended for educational purposes and should not be used as the sole basis for actual financial decisions.

---

## 🚀 14. Future Improvements

- Use larger and more diverse credit card application datasets.
- Experiment with additional Machine Learning algorithms.
- Improve model performance through hyperparameter tuning.
- Apply cross-validation for more reliable evaluation.
- Add more data visualization features.
- Improve the Streamlit web application's interface.
- Add batch prediction functionality.
- Deploy the application on a cloud platform.
- Perform further testing using new application records.

---

## 🏁 15. Conclusion

This project demonstrates the practical application of Machine Learning for credit card application classification.

It includes data inspection, preprocessing, Exploratory Data Analysis, model training, performance comparison, model testing, and a Streamlit web application.

By implementing Support Vector Machine, K-Nearest Neighbors, and Gradient Boosting, the project provides an opportunity to compare three different classification algorithms.

The best-performing model is selected based on evaluation results and integrated into the Streamlit application.

Overall, this project provides practical experience in Python programming, data analysis, Machine Learning algorithms, model evaluation, and web application development.

---

## 📚 16. Dataset Reference

**Dataset Name:** Credit Card Applications Dataset

**Dataset File:** `Credit_Card_Applications.csv`

**Target Variable:** `Class`

**Input Features:** `A1` to `A14`

The dataset is used for educational Machine Learning classification and model comparison.

---

<div align="center">

## 👨‍💻 Project Information

**Project Title:** Credit Card Application Prediction Using Machine Learning

**Developer:** Moloy Malik

**GitHub Repository:** [Credit Card Application Prediction](https://github.com/MoloyMalik/Credit_Card_Application_Prediction)

💳 Financial Data Analysis • 🤖 Machine Learning • 📊 Smart Classification

</div>

---

## ⭐ Support the Project

If you find this project useful, please consider giving the repository a **Star ⭐** on GitHub.

Your support and feedback are appreciated!

#MachineLearning #Python #CreditCardApplicationPrediction #ArtificialIntelligence #DataScience #SVM #KNN #GradientBoosting #ScikitLearn #Streamlit #Classification #EDA #PredictiveAnalytics #MachineLearningProject #GitHub
