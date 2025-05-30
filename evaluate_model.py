import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

print("Loading model and test data...")
try:
    # Load the model, scaler, and test data
    model = joblib.load("/home/ubuntu/logistic_regression_model.joblib")
    X_test_scaled = np.load("/home/ubuntu/X_test_scaled.npy")
    y_test = np.load("/home/ubuntu/y_test.npy")
    label_encoder = joblib.load("/home/ubuntu/label_encoder.joblib") # Needed for class labels in plot
    print("Model and data loaded successfully.")

    print("\nEvaluating model...")
    # 1. Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class (Malignant)

    # 2. Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(cm)
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    cm_plot_path = "/home/ubuntu/confusion_matrix.png"
    plt.savefig(cm_plot_path)
    print(f"Confusion Matrix plot saved to {cm_plot_path}")
    plt.close(fig)

    # 3. Calculate Precision, Recall, F1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 4. Calculate ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 5. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    roc_plot_path = "/home/ubuntu/roc_curve.png"
    plt.savefig(roc_plot_path)
    print(f"ROC curve plot saved to {roc_plot_path}")
    plt.close(fig)

    # Save evaluation results to a file
    eval_summary_path = "/home/ubuntu/evaluation_summary.txt"
    with open(eval_summary_path, 'w') as f:
        f.write("Model Evaluation Summary\n")
        f.write("========================\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write(f"\nTrue Negatives (TN): {tn}")
        f.write(f"\nFalse Positives (FP): {fp}")
        f.write(f"\nFalse Negatives (FN): {fn}")
        f.write(f"\nTrue Positives (TP): {tp}")
        f.write(f"\n\nPrecision: {precision:.4f}")
        f.write(f"\nRecall: {recall:.4f}")
        f.write(f"\nF1-Score: {f1:.4f}")
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"\nEvaluation summary saved to {eval_summary_path}")

    print("\nModel evaluation complete.")

except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}. Please ensure model and data files exist.")
except Exception as e:
    print(f"An error occurred during model evaluation: {e}")


