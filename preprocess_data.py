import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset again (or pass the dataframe if running sequentially)
column_names = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]
data_path = '/home/ubuntu/Downloads/wdbc.data'
df = pd.read_csv(data_path, header=None, names=column_names)

print("Preprocessing data...")

# 1. Separate features (X) and target variable (y)
# Drop the ID column as it's not a feature
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode the target variable (M=1, B=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Target variable encoded. Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}") # M=1, B=0

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Data split into train/test sets.")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# 3. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized using StandardScaler.")

# Optional: Save the processed data (e.g., as numpy arrays or csv)
# This is useful if the next steps are in separate scripts
import numpy as np
np.save('/home/ubuntu/X_train_scaled.npy', X_train_scaled)
np.save('/home/ubuntu/X_test_scaled.npy', X_test_scaled)
np.save('/home/ubuntu/y_train.npy', y_train)
np.save('/home/ubuntu/y_test.npy', y_test)
print("Processed data saved to .npy files.")

# Save the scaler object for later use (e.g., if deploying the model)
import joblib
joblib.dump(scaler, '/home/ubuntu/scaler.joblib')
joblib.dump(label_encoder, '/home/ubuntu/label_encoder.joblib')
print("Scaler and Label Encoder saved.")

print("\nPreprocessing complete.")

