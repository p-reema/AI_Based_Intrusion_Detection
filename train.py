import os
import joblib
# import mlflow
# import mlflow.sklearn
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from src.model import get_model
import matplotlib.pyplot as plt
from utils.evaluate import evaluate_model,shap_explainer
from xgboost import plot_importance
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# from sklearn.metrics import roc_auc_score, accuracy_score
# from mlflow.models.signature import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Create a folder named "outputs" (or any name you prefer)
os.makedirs("outputs", exist_ok=True)
os.makedirs("pickle", exist_ok=True)


# Load data
data = pd.read_csv(r'C:\Users\Preema S\Desktop\AI-Intrusion-Detection\dataset\intrusion_detection_dataset.csv')
# Optional: Sample numeric columns for visualization
numeric_cols = data.select_dtypes(include='number').columns

# Boxplot before clipping
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numeric_cols])
plt.title("Before Outlier Clipping")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/Before_Outlier_Clipping.jpg')
plt.show()

# Perform clipping (as before)
for col in data.select_dtypes(include='number').columns:
    lower = data[col].quantile(0.01)
    upper = data[col].quantile(0.99)
    data[col] = data[col].clip(lower, upper)

# Boxplot after clipping
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numeric_cols])
plt.title("After Outlier Clipping")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/After_Outlier_Clipping.jpg')
plt.show()

# Assume last column is label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1234)

# Fit on training data, transform both train and test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'pickle/Standard_Scaler.pkl')

# Get model
model = get_model()
model.fit(X_train, y_train)  # No early_stopping here!

# Use best_estimator_ if GridSearchCV
best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model

# Train(if gridsearchCV not used)
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_test, y_test)],
#     early_stopping_rounds=10,
#     verbose=True
# )

# Evaluate the model
evaluate_model(model, X_test, y_test)
shap_explainer(model,X_test)

# Plot top 10 features
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig('outputs/XGBoost_Feature_Importance.jpg')
plt.show()


# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/Confusion_Matrix.jpg")
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('outputs/ROC_Curve.jpg')
plt.show()


# pickle file
joblib.dump(model, 'pickle/xgboost_intrusion_model.pkl')
