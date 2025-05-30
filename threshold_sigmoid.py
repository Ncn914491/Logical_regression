import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt
import os

print("Loading model and test data for threshold analysis...")
try:
    # Load the model, test data, and probabilities
    model = joblib.load("/home/ubuntu/logistic_regression_model.joblib")
    X_test_scaled = np.load("/home/ubuntu/X_test_scaled.npy")
    y_test = np.load("/home/ubuntu/y_test.npy")
    # Regenerate probabilities as they weren't saved explicitly
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    print("Model and data loaded successfully.")

    # --- Threshold Tuning --- 
    print("\nAnalyzing precision-recall trade-off with different thresholds...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Plot Precision-Recall vs Threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    # Exclude the last threshold which corresponds to recall=0
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs. Decision Threshold")
    ax.legend(loc="center left")
    ax.grid(True)
    threshold_plot_path = "/home/ubuntu/precision_recall_threshold.png"
    plt.savefig(threshold_plot_path)
    print(f"Precision-Recall vs Threshold plot saved to {threshold_plot_path}")
    plt.close(fig)

    # Example: Find threshold for a specific recall (e.g., 0.90)
    # Note: This is just an example demonstration
    try:
        # Find the first threshold where recall is >= 0.90
        idx = np.where(recalls[:-1] >= 0.90)[0]
        if len(idx) > 0:
            threshold_for_recall_90 = thresholds[idx[0]]
            print(f"Example: Threshold for Recall >= 0.90: ~{threshold_for_recall_90:.2f}")
            # Calculate precision at this threshold
            y_pred_custom_thresh = (y_pred_proba >= threshold_for_recall_90).astype(int)
            precision_at_thresh = precision_score(y_test, y_pred_custom_thresh)
            recall_at_thresh = recall_score(y_test, y_pred_custom_thresh)
            print(f"  Precision at this threshold: {precision_at_thresh:.4f}")
            print(f"  Recall at this threshold: {recall_at_thresh:.4f}")
        else:
            print("Could not find a threshold where recall >= 0.90")
    except Exception as ve:
        print(f"Error finding threshold for recall 0.90: {ve}")

    # --- Sigmoid Function --- 
    print("\nPlotting the Sigmoid function...")
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z = np.linspace(-10, 10, 100)
    s = sigmoid(z)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, s)
    ax.set_xlabel("z (Linear Combination Input)")
    ax.set_ylabel("Sigmoid(z) (Probability)")
    ax.set_title("Sigmoid Function")
    ax.grid(True)
    ax.axhline(0.5, color='grey', linestyle='--') # Corrected quotes
    ax.axvline(0, color='grey', linestyle='--')   # Corrected quotes
    sigmoid_plot_path = "/home/ubuntu/sigmoid_function.png"
    plt.savefig(sigmoid_plot_path)
    print(f"Sigmoid function plot saved to {sigmoid_plot_path}")
    plt.close(fig)

    print("\nThreshold tuning analysis and Sigmoid plot generation complete.")

except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}. Please ensure model and data files exist.")
except Exception as e:
    print(f"An error occurred: {e}")


