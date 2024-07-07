from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from utils import preprocess_image

# Load the trained model
model = tf.keras.models.load_model('model/model.h5')

app = Flask(__MedLens__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded files
    eye_image = request.files['eye_image']
    tongue_image = request.files['tongue_image']
    
    # Get the health parameters
    age = float(request.form['age'])
    weight = float(request.form['weight'])
    blood_pressure = float(request.form['blood_pressure'])
    
    # Preprocess the images
    eye_image = preprocess_image(eye_image, target_size=(64, 64))
    tongue_image = preprocess_image(tongue_image, target_size=(64, 64))
    
    # Prepare the health parameters
    health_params = np.array([[age, weight, blood_pressure]])
    
    # Predict using the model
    prediction = model.predict([eye_image, tongue_image, health_params])
    health_status = 'Healthy' if prediction[0][0] > 0.5 else 'Unhealthy'
    
    return jsonify({
        'health_status': health_status,
        'prediction_score': float(prediction[0][0])
    })

if __name__ == '__main__':
    app.run(debug=True)
