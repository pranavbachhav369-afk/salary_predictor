from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route (browser test)
@app.route("/")
def home():
    return "API is running successfully"

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # input features
        features = data["features"]

        # convert to numpy array
        input_data = np.array(features).reshape(1, -1)

        # prediction
        prediction = model.predict(input_data)

        return jsonify({
            "success": True,
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Render / production support
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)