import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load the dataset (assuming it's already in a DataFrame `df`)
# If not, load from CSV: df = pd.read_csv("your_dataset.csv")
df = pd.read_csv("Synthetic_Breast_Cancer_Dataset.csv")

# --- 2. Encode categorical features ---
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
categorical_cols.remove("diagnosis")  # Target column

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
df["diagnosis"] = target_encoder.fit_transform(df["diagnosis"])  # Positive=1, Negative=0

# --- 3. Split into features and target ---
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Train Random Forest Model ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = rf_model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# --- 7. Save model and encoders ---
joblib.dump(rf_model, "rf_breast_cancer_model.joblib")
joblib.dump(encoders, "rf_feature_encoders.joblib")
joblib.dump(target_encoder, "rf_target_encoder.joblib")

print("✅ Model and encoders saved.")


from sklearn.impute import KNNImputer

# Train imputer on full feature matrix
imputer = KNNImputer(n_neighbors=5)
imputer.fit(X)

# Save the imputer
joblib.dump(imputer, "rf_feature_imputer.joblib")


# After training
joblib.dump(rf_model, "rf_model.joblib")
joblib.dump(imputer, "knn_imputer.joblib")
joblib.dump(X.columns.tolist(), "feature_names.joblib")

# Get top N features (e.g., 7)
importances = rf_model.feature_importances_
top_features = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(7).index.tolist()
joblib.dump(top_features, "important_features.joblib")

