# train_model.py
# Place this file in your project root directory (road_safety_ai/)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'prediction', 'ml', 'accidents.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'prediction', 'ml_model.pkl')

print("=" * 60)
print("ROAD SAFETY AI - MODEL TRAINING")
print("=" * 60)

# Step 1: Load Data
print("\n[1/5] Loading data from:", DATA_PATH)
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"✖ Error: File not found at {DATA_PATH}")
    print("\nPlease ensure 'accidents.csv' exists in prediction/ml/ directory")
    exit(1)

# Step 2: Define Features
print("\n[2/5] Preparing features...")

# These are the 11 features your model expects (in this exact order)
FEATURE_COLUMNS = [
    'Did_Police_Officer_Attend_Scene_of_Accident',
    'Age_of_Driver',
    'Vehicle_Type',
    'Age_of_Vehicle',
    'Engine_Capacity_(CC)',
    'Day_of_Week',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Light_Conditions',
    'Sex_of_Driver',
    'Speed_limit'
]

TARGET_COLUMN = 'Accident_Severity'  # Change this if your target column has a different name

# Check if all columns exist
missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
if missing_cols:
    print(f"✖ Missing columns in CSV: {missing_cols}")
    print(f"\nAvailable columns: {list(df.columns)}")
    print("\nPlease update FEATURE_COLUMNS or TARGET_COLUMN to match your CSV")
    exit(1)

# Extract features and target
X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

# Step 3: Handle Missing Values
print("\n[3/5] Handling missing values...")
print(f"Missing values before: {X.isnull().sum().sum()}")

# Fill missing values with median for numeric columns
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"  - Filled {col} with median: {median_val}")

# Remove rows with missing target
initial_len = len(y)
mask = y.notnull()
X = X[mask]
y = y[mask]
print(f"✓ Removed {initial_len - len(y)} rows with missing target")
print(f"✓ Final dataset size: {len(X)} records")

# Step 4: Train Model
print("\n[4/5] Training Random Forest model...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  - Training set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle imbalanced classes
)

print("  - Training in progress...")
model.fit(X_train, y_train)
print("✓ Training complete!")

# Step 5: Evaluate Model
print("\n[5/5] Evaluating model performance...")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Accuracy: {accuracy:.2%}")

# Show classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['FATAL', 'SERIOUS', 'SLIGHT']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save Model
print(f"\n[SAVING] Saving model to: {MODEL_PATH}")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("✓ Model saved successfully!")

# Verify saved model
print("\n[VERIFICATION] Testing saved model...")
loaded_model = joblib.load(MODEL_PATH)
test_prediction = loaded_model.predict(X_test[:1])
print(f"✓ Model loads correctly. Test prediction: {test_prediction[0]}")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel file: {MODEL_PATH}")
print(f"Accuracy: {accuracy:.2%}")
print("\nYou can now use the prediction API at: /predict_json/")
print("=" * 60)