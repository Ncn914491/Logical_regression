import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading preprocessed training data...")
try:
    X_train_scaled = np.load("/home/ubuntu/X_train_scaled.npy")
    y_train = np.load("/home/ubuntu/y_train.npy")
    print("Data loaded successfully.")

    print("\nTraining Logistic Regression model...")
    # Initialize the model
    log_reg_model = LogisticRegression(random_state=42)

    # Train the model
    log_reg_model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # Save the trained model
    joblib.dump(log_reg_model, "/home/ubuntu/logistic_regression_model.joblib")
    print(f"Trained model saved to /home/ubuntu/logistic_regression_model.joblib")

except FileNotFoundError:
    print("Error: Preprocessed data files not found. Please run the preprocessing script first.")
except Exception as e:
    print(f"An error occurred during model training: {e}")

