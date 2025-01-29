import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize backend bridge
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://darttgoblin.github.io"}})

# Dictionary to map numeric values to emotion words
emotions = {
    0: 'Sadness 😢',
    1: 'Joy 😊',
    2: 'Love ❤️',
    3: 'Anger 😡',
    4: 'Fear 😨',
}

# Load the trained model
with open('Cortext.pkl', 'rb') as pipeline_file:
    cortext = pickle.load(pipeline_file)

@app.route('/', methods=['POST'])
def handle_request():
    data = request.get_json()
    text = data.get('text')
    
    emotion_index = cortext.predict([text])[0]
    emotion = emotions.get(emotion_index)
    probabilities = cortext.predict_proba([text])[0]
    normalized_probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

    response = {
        'success': True,
        'emotion': emotion,
        'confidence': list(normalized_probabilities)
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))  # Railway's assigned port
    app.run(host='0.0.0.0', port=port)
