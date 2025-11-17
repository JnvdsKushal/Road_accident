import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = "ml/accidents.csv"  # change this to your exact file name
print(f"📂 Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Show initial info
print("\n🔍 Initial data info:")
print(df.info())

# Drop columns that are entirely empty (if any)
df.dropna(axis=1, how='all', inplace=True)

# Fill missing numerical values with 0
df.fillna(0, inplace=True)

# Encode 'State/UT/City' or 'Category' if needed
if 'State/UT/City' in df.columns:
    le_state = LabelEncoder()
    df['State/UT/City'] = le_state.fit_transform(df['State/UT/City'])

if 'Category' in df.columns:
    le_cat = LabelEncoder()
    df['Category'] = le_cat.fit_transform(df['Category'])

# Remove commas or spaces in column names
df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_', regex=True)

# Handle potential duplicates
df.drop_duplicates(inplace=True)

# Normalize numeric columns for ML training
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save cleaned dataset
output_path = "ml/cleaned_accident_data.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned data saved at: {output_path}")
print(f"🧾 Total Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n💡 Ready for model training!")
