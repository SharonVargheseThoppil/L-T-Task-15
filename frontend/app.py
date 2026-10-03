import os

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "cifar10_cnn_model.keras")
)
APP_VERSION = os.environ.get(
    "APP_VERSION",
    "v2.0"
)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((32, 32))

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "application": "CIFAR-10 Deep Learning API",
        "version": APP_VERSION,
        "status": "running",
        "endpoints": [
            "/",
            "/health",
            "/predict"
        ]
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "version": APP_VERSION
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image provided",
            "message": "Upload an image using the 'image' field"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename",
            "message": "Please select an image file"
        }), 400

    try:

        image = Image.open(file.stream)

        processed_image = preprocess_image(image)

        predictions = model.predict(
            processed_image,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[0][predicted_index]
        )

        probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(
                confidence * 100,
                2
            ),
            "class_index": predicted_index,
            "probabilities": probabilities,
            "version": APP_VERSION
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(error)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
