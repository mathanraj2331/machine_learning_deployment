import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained logistic regression model and scaler
# These must be loaded before any functions or routes that depend on them
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

print("Model and scaler loaded successfully in app.py")

# Define the numerical and categorical columns as used during training
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# The exact columns the model expects after preprocessing
# This list should be obtained from X_train_scaled.columns
# It is crucial to maintain the order and presence of all columns
# For simplicity, hardcoding it based on previous notebook steps.
model_columns = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_1', 'cp_1', 'cp_2',
    'cp_3', 'fbs_1', 'restecg_1', 'restecg_2', 'exang_1', 'slope_1', 'slope_2',
    'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_1', 'thal_2', 'thal_3'
]

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocesses raw input data for the logistic regression model.

    Args:
        data (dict): A dictionary containing raw input features.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for model prediction.
    """
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([data])

    # Apply one-hot encoding to categorical features
    # drop_first=True to match the training preprocessing
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Reindex the DataFrame to match the model_columns
    # This ensures that all expected columns are present, and in the correct order
    # Missing columns will be added with 0, extra columns will be dropped
    input_preprocessed = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Apply StandardScaler to numerical features
    input_preprocessed[numerical_cols] = scaler.transform(input_preprocessed[numerical_cols])

    return input_preprocessed

@app.route('/')
def hello():
    return 'Flask app is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        try:
            # Preprocess the input data
            processed_data = preprocess_input(data)

            # Make prediction
            prediction = model.predict(processed_data)

            # Convert prediction to Python int (numpy int is not directly JSON serializable)
            prediction_result = int(prediction[0])

            return jsonify({'prediction': prediction_result})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

print("app.py created successfully with all components.")
