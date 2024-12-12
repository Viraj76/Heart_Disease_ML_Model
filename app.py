from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load the trained model and preprocessing mappings
model = joblib.load('heart_model.pkl')
mappings = joblib.load('heart_preprocessing_mappings.pkl')

app = Flask(__name__)
CORS(app)

def preprocess_input(data):
    """Preprocess input data using the mappings."""
    data['BMI Category'] = mappings['bmi_mapping'].get(data['BMI Category'], -1)
    data['Occupation'] = mappings['occupation_mapping'].get(data['Occupation'], -1)
    data['Gender'] = mappings['gender_mapping'].get(data['Gender'], -1)
    defaults = {
        'Age': 0,
        'BMI Category': -1,
        'Occupation': -1,
        'Gender': -1,
        'Systolic': 120,
        'Diastolic': 80,
        'Heart Rate': 70,
        'Stress Level': 5,
        # Add placeholder values for any additional features
        'Feature1': 0,
        'Feature2': 0,
        'Feature3': 0,
        'Feature4': 0,
        'Feature5': 0,
    }

    # Merge defaults with provided data
    for key, value in defaults.items():
        data[key] = data.get(key, value)
    return data

@app.route('/')
def home():
    """Health check endpoint."""
    return "ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions."""
    try:
        # Get JSON data from the request
        input_data = request.json

        # Preprocess the input
        processed_data = preprocess_input(input_data)

        # Extract features for the model
        features = [
            processed_data['Age'],
            processed_data['BMI Category'],
            processed_data['Occupation'],
            processed_data['Gender'],
            processed_data['Systolic'],
            processed_data['Diastolic'],
            processed_data['Heart Rate'],
            processed_data['Stress Level'],
            processed_data['Feature1'],
            processed_data['Feature2'],
            processed_data['Feature3'],
            processed_data['Feature4'],
            processed_data['Feature5']
        ]

        # Make prediction
        prediction = model.predict([features])[0]

        # Return the result
        return jsonify({'Heart Disease': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

