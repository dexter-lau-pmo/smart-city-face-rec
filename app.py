from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('keras_model.h5', compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image):
    # Resize image to match model input shape
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    # Convert image to array
    image_array = np.asarray(image)
    # Normalize pixel values
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Expand dimensions to match model input
    return np.expand_dims(normalized_image_array, axis=0)


def predict_image(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])
        return {"class": class_name[2:-1], "confidence": confidence_score}
    except Exception as e:
        return {"error": str(e)}


@app.route('/validate', methods=['POST'])
def validate_images():
    try:
        data = request.get_json(force=True)
        results = {}
        for url in data['images']:
            result = predict_image(url)
            results[url] = result
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/ping', methods=['GET'])
def ping():
    try:
        return "", 200
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
