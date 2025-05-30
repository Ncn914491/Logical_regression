import pandas as pd

# Define column names based on wdbc.names
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32) Ten real-valued features are computed for each cell nucleus:
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)
# The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

column_names = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Load the dataset
data_path = '/home/ubuntu/Downloads/wdbc.data'
try:
    df = pd.read_csv(data_path, header=None, names=column_names)

    print("Dataset loaded successfully.")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Info:")
    df.info()

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nTarget variable (diagnosis) distribution:")
    print(df['diagnosis'].value_counts())

    # Save exploration results to a file
    with open('/home/ubuntu/data_exploration_summary.txt', 'w') as f:
        f.write("Dataset Exploration Summary\n")
        f.write("=============================\n")
        f.write("\nFirst 5 rows:\n")
        f.write(df.head().to_string())
        f.write("\n\nDataset Info:\n")
        # Capture df.info() output
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("\n\nMissing values per column:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\nTarget variable (diagnosis) distribution:\n")
        f.write(df['diagnosis'].value_counts().to_string())

    print("\nExploration summary saved to /home/ubuntu/data_exploration_summary.txt")

except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
except Exception as e:
    print(f"An error occurred: {e}")


