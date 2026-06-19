from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# Flask app
app = Flask(__name__)

# Enable CORS (IMPORTANT for Lovable frontend)
CORS(app)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route (for testing)
@app.route("/")
def home():
    return "API is running successfully 🚀"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # validate input
        if "features" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'features' key"
            })

        features = data["features"]

        # convert to numpy array
        input_data = np.array(features).reshape(1, -1)

        # prediction
        prediction = model.predict(input_data)

        return jsonify({
            "success": True,
            "prediction": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Run app (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)