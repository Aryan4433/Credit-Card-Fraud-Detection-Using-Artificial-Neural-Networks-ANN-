
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model and scaler
model = load_model('fraud_detection_model.h5')
scaler = joblib.load('scaler.save')

@app.route('/')
def home():
    return "🚀 Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)  # Expect 30 features
        scaled_features = scaler.transform(features)
        prob = model.predict(scaled_features)[0][0]
        threshold = 0.3
        label = int(prob > threshold)
        return jsonify({
            'fraud_probability': float(prob),
            'prediction': 'FRAUD' if label == 1 else 'LEGIT'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
