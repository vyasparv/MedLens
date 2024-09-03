from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Load models
eye_model = load_model('models/eye_disease_model.keras')
tongue_model = load_model('models/tongue_disease_model.keras')

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    
    target = data['target']
    if target == 'eye':
        model = eye_model
    elif target == 'tongue':
        model = tongue_model
    else:
        return jsonify({'error': 'Invalid target'}), 400

    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image).tolist()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
