import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Drop irrelevant columns
df.drop(columns=['sl_no', 'salary'], inplace=True)

# Fill missing numerical values with mean
numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Drop rows with missing target label
df = df[df['status'].notna()]

# Encode categorical columns
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(columns=['status'])
y = df['status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("placement_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metadata
with open("model_metadata.pkl", "wb") as f:
    pickle.dump({'features': X.columns.tolist(), 'label_encoders': label_encoders}, f)

print("✅ Model trained and saved successfully.")
print("✅ Model trained with features:")
print(X.columns.tolist())  # <-- Add this line

print("Training features:", X.columns.tolist())
