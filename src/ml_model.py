# ml_model.py

import pandas as pd
import joblib
from sklearn.impute import KNNImputer

# Load trained model & imputer
model = joblib.load("/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/src/weights/rf_model.joblib")
imputer = joblib.load("/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/src/weights/knn_imputer.joblib")
feature_names = joblib.load("/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/src/weights/feature_names.joblib")  # All expected features
important_features = joblib.load("/media/husseinmleng/New Volume/Jupyter_Notebooks/Freelancing/Breast-Cancer/src/weights/important_features.joblib")  # Top features

def predict_cancer_risk(user_input: dict):
    print("🔍 Predicting cancer risk with user input:", user_input)
    print("Using features:", important_features)
    print("Starting prediction...")
    # Initialize all features with None
    full_input = {f: None for f in feature_names}
    
    # Fill known values
    for key, value in user_input.items():
        full_input[key] = value

    # Convert to DataFrame
    input_df = pd.DataFrame([full_input])

    # Impute missing values
    input_df_imputed = pd.DataFrame(
        imputer.transform(input_df),
        columns=feature_names
    )

    # Predict
    probs = model.predict_proba(input_df_imputed)[0]
    prediction = model.predict(input_df_imputed)[0]
    confidence = max(probs)

    return prediction, confidence
