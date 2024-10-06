from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

# Disable oneDNN custom operations and suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the class labels for your custom model
class_labels = ['EarlyPreB', 'PreB', 'ProB', 'Benign']

# Define a Flask app
app = Flask(__name__)

# Define model paths
MODEL_PATHS = {
    'EfficientNetB0': 'models/EfficientNetB0.tflite',
    'MobileNetV2': 'models/MobileNetV2.tflite',
    'NasNetMobile': 'models/NasNetMobile.tflite'
}

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load the TFLite model and allocate tensors
def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if model_path is None:
        raise ValueError(f"Model '{model_name}' not found.")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess the image and predict the class using the TFLite model
def model_predict(img_path, interpreter):
    try:
        # Load the image with the target size
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

        # Set the tensor to the input of the model
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        preds = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

        return preds
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

# API to get available models
@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(list(MODEL_PATHS.keys()))

# API to handle image uploads and predictions
@app.route('/api/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part."}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No selected file."}), 400

    selected_model = request.form.get('model')
    if selected_model not in MODEL_PATHS:
        return jsonify({"error": "Invalid model selected."}), 400

    # Save the file to the uploads directory
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    # Load the model and make a prediction using the uploaded image
    try:
        interpreter = load_model(selected_model)
        preds = model_predict(file_path, interpreter)

        if preds is None:
            return jsonify({"error": "Prediction failed."}), 500

        # Get the index of the class with the highest probability
        pred_class_idx = np.argmax(preds, axis=1)[0]

        # Return the predicted class as the result
        result = class_labels[pred_class_idx] if pred_class_idx < len(class_labels) else "Prediction index out of range."
        return jsonify({"predicted_class": result})

    except Exception as e:
        print(f"Error during upload or prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
