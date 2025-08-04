import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
import joblib
from pathlib import Path

def train_and_save_model():
    """
    Loads data, trains a Random Forest model, and saves all necessary artifacts.
    """
    # --- 1. Setup Paths ---
    # Create paths relative to this script's location
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / "Synthetic_Breast_Cancer_Dataset.csv" # Assuming dataset is in the root
    weights_dir = current_dir / "weights"
    weights_dir.mkdir(exist_ok=True) # Ensure the directory exists

    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    # --- 3. Encode Categorical Features ---
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    if "diagnosis" in categorical_cols:
        categorical_cols.remove("diagnosis")

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode target column
    target_encoder = LabelEncoder()
    df["diagnosis"] = target_encoder.fit_transform(df["diagnosis"])

    # --- 4. Split Features and Target ---
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # --- 5. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 6. Train KNN Imputer ---
    # Train imputer on the full feature matrix to learn relationships
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(X)

    # --- 7. Train Random Forest Model ---
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # --- 8. Evaluate ---
    y_pred = rf_model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    # --- 9. Get Top Features ---
    importances = rf_model.feature_importances_
    top_features = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(7).index.tolist()
    print(f"\nTop 7 important features: {top_features}")

    # --- 10. Save All Artifacts ---
    joblib.dump(rf_model, weights_dir / "rf_model.joblib")
    joblib.dump(imputer, weights_dir / "knn_imputer.joblib")
    joblib.dump(X.columns.tolist(), weights_dir / "feature_names.joblib")
    joblib.dump(top_features, weights_dir / "important_features.joblib")
    joblib.dump(encoders, weights_dir / "rf_feature_encoders.joblib")
    joblib.dump(target_encoder, weights_dir / "rf_target_encoder.joblib")

    print(f"\n✅ Model and all artifacts saved successfully to '{weights_dir}'.")

if __name__ == "__main__":
    train_and_save_model()