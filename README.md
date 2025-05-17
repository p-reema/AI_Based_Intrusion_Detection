# AI-Based Intrusion Detection System using XGBoost

This project implements a machine learning-based Intrusion Detection System (IDS) using the XGBoost algorithm. It includes preprocessing, outlier handling, feature scaling, model training, evaluation, and saving the model for future inference.

---

## 📁 Project Structure

```
├── dataset/
│   └── intrusion_detection_dataset.csv
├── outputs/
│   ├── Before_Outlier_Clipping.jpg
│   ├── After_Outlier_Clipping.jpg
│   ├── XGBoost_Feature_Importance.jpg
│   ├── Confusion_Matrix.jpg
│   └── ROC_Curve.jpg
├── pickle/
│   ├── xgboost_intrusion_model.pkl
│   └── standard_scaler.pkl
├── src/
│   └── model.py             # Function to return XGBoost model
├── utils/
│   └── evaluate.py          # Model evaluation function
├── main.py                  # Main training script
└── README.md
```

---

## 📊 Dataset

* **Source** : `intrusion_detection_dataset.csv`
* Contains network traffic data labeled as either **normal(0)** or  **intrusion(1)** .
* Numeric features describing various characteristics of the traffic.

---

## 🔁 Workflow Overview

### 1. Data Preprocessing

* Load the dataset
* Handle outliers using clipping (1st and 99th percentile)
* Visualize outliers before and after clipping
* Standardize features using `StandardScaler`

### 2. Model Training

* Split dataset (80/20 train-test)
* Train XGBoost classifier with early stopping
* Save the trained model as a `.pkl` file

### 3. Model Evaluation

* Evaluate model using:
  * ROC Curve & AUC Score (achieved **0.97 AUC**)
  * Confusion Matrix
  * Top 10 most important features from XGBoost

---

## 📈 Results

| Metric             | Value               |
| ------------------ | ------------------- |
| AUC Score          | **0.97**      |
| Confusion Matrix   | ✔️ Saved as image |
| Feature Importance | ✔️ Top 10 plotted |

---

## 📦 Dependencies

Install required libraries via pip:

```
pip install -r requirements.txt
```

Example requirements:

```
pandas
scikit-learn
xgboost
seaborn
matplotlib
joblib
```

---

## 💾 Model Saving

* Model: `pickle/xgboost_intrusion_model.pkl`
* Scaler: `pickle/standard_scaler.pkl`

Use `joblib.load()` to reuse them during inference.

---

## 🚀 Run the Project

```
python train.py
```

---

## ✅ 

## ✅ Advanced Topics & Future Improvements

### 1. SHAP for Model Explainability

* Use **SHAP (SHapley Additive exPlanations)** to interpret model predictions.
* Helps understand feature impact on individual predictions.
* Save SHAP plots as artifacts for better model explainability.

---

### 2. Hyperparameter Tuning with GridSearchCV

* Use `GridSearchCV` to find the best hyperparameters automatically.
* Helps improve model performance by exhaustive search over specified parameter grid.
* Integrate into pipeline before final model training.

---

### 3. Handling Imbalanced Data with SMOTE

* Use **SMOTE (Synthetic Minority Over-sampling Technique)** to balance imbalanced datasets.
* Generates synthetic samples for minority class to improve classifier training.
* Apply **before** model training, after train-test split.

## ✅ Future Improvements

### Experiment Tracking with MLflow

* Track models, parameters, metrics, and artifacts automatically.
* Log SHAP plots, GridSearch results, and SMOTE effects for reproducibility.
