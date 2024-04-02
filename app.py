from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

class_names = ["Dry Skin", "Combination Skin", "Oily Skin", "Normal Skin", "Sensitive Skin"]

def preprocess_image(image):
    img = image.resize((150, 150))
    img = np.array(img, dtype=np.float32) / 255.0  # Convert to FLOAT32
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/predict', methods=['POST'])
def predict_skin_type():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    try:
        img = Image.open(image)
        img_array = preprocess_image(img)

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on input data.
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_prob_percentage = {class_names[i]: round(prob * 100, 2) for i, prob in enumerate(output_data[0])}
        return jsonify(pred_prob_percentage)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
