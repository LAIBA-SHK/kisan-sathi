import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the separate training and testing datasets
train_df = pd.read_csv("crop_data/train.csv")
test_df = pd.read_csv("crop_data/test.csv")

# Load the ORIGINAL full dataset to get the original crop names
# This is a one-off action to get the list of unique crop names
original_df = pd.read_csv("crop_data/Crop_recommendation.csv")
original_crop_names = sorted(original_df['label'].unique())

# Separate features (X) and target (y) for both datasets
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Since your labels are already integers, you don't need to fit a LabelEncoder here.
# You just need to transform the integer labels to ensure they start from 0 if they don't already.
# For simplicity, we'll assume they are correct and just use the original names.
# (If they aren't, you would need to adjust this logic)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation - using the original string names you loaded
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=original_crop_names))

# Save the trained model
joblib.dump(model, "crop_recommendation_model.pkl")

# If you need to save an encoder, save it from the load_dataset.py file.
# The label encoder mapping should be a part of the data preparation step, not training.
# You can load it here if needed for prediction later.
# joblib.dump(le, "label_encoder.pkl")
# print("✅ Model and encoder saved successfully!")