from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

with open("Cortext.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Prediction
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        # Convert numpy types to native Python
        prediction = str(prediction)
        probabilities = {str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)}

        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)