
import pandas as pd
import joblib
import os

# --- FIX: Use relative paths instead of absolute paths ---
# Get the directory where the current script is located
_script_dir = os.path.dirname(__file__)
_weights_dir = os.path.join(_script_dir, "weights")

# Load trained model & imputer using robust relative paths
model_path = os.path.join(_weights_dir, "rf_model.joblib")
imputer_path = os.path.join(_weights_dir, "knn_imputer.joblib")
feature_names_path = os.path.join(_weights_dir, "feature_names.joblib")
important_features_path = os.path.join(_weights_dir, "important_features.joblib")

model = joblib.load(model_path)
imputer = joblib.load(imputer_path)
feature_names = joblib.load(feature_names_path)
important_features = joblib.load(important_features_path)

def predict_cancer_risk(user_input: dict):
    """Predicts cancer risk based on user input after imputation."""
    print("🔍 Predicting cancer risk with user input:", user_input)
    
    # Create a DataFrame with the exact structure the model expects
    # Initialize all features with None (or np.nan)
    full_input = {f: None for f in feature_names}
    
    # Fill in the values we received from the user
    for key, value in user_input.items():
        if key in full_input:
            full_input[key] = value

    # Convert to DataFrame
    input_df = pd.DataFrame([full_input], columns=feature_names)

    # Impute missing values for features that were not provided by the user
    input_df_imputed = pd.DataFrame(
        imputer.transform(input_df),
        columns=feature_names
    )

    # Select only the important features for prediction, in the correct order
    # This step is crucial if the model was trained only on these
    # If the model was trained on all features, this line can be removed.
    # Assuming the model was trained on all features after imputation.
    final_input_df = input_df_imputed

    # Predict probabilities and the final outcome
    probs = model.predict_proba(final_input_df[model.feature_names_in_])[0]
    prediction = model.predict(final_input_df[model.feature_names_in_])[0]
    confidence = max(probs)

    print(f"Prediction: {prediction}, Confidence: {confidence}")
    return prediction, confidence
