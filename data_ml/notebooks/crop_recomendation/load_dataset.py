import zipfile
import pandas as pd

# Unzip and load CSV
with zipfile.ZipFile("crop-recommendation-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("crop_data")

df = pd.read_csv("crop_data/Crop_recommendation.csv")

# Check first rows
print(df.head())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Show shape
print("Rows, Columns:", df.shape)
from sklearn.preprocessing import LabelEncoder

# Convert crop names to numeric labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

print("First 5 rows after encoding labels:")
print(df.head())
# Separate input features and target
X = df.drop('label', axis=1)  # all columns except label
y = df['label']               # target column
from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
# Assuming you have already split the data as shown in your code
# Combine features and target for saving to CSV
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save the new dataframes as CSV files
# Make sure the data_ml/datasets/crop_recommendation/ directories exist
train_df.to_csv('crop_data/train.csv', index=False)
test_df.to_csv('crop_data/test.csv', index=False)

print("Dataset split successfully and saved to train.csv and test.csv")
import joblib
joblib.dump(le, "label_encoder.pkl")