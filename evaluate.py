import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# Create SHAP explainer
def shap_explainer(model,X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot (feature impact)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)

    # Save plot as image (optional)
    plt.savefig('outputs/shap_summary_plot.png',bbox_inches='tight')
    plt.show()

