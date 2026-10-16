
from pathlib import Path
import json
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
FRONTEND_DIR = ROOT / "frontend"

MODEL_PATH = Path(os.environ.get(
    "TOMATODOC_MODEL",
    str(AI_DIR / "models" / "tomatodoc_mobilenetv2_tuned_v4_best.keras")
))

LABELS_PATH = Path(os.environ.get(
    "TOMATODOC_LABELS",
    str(AI_DIR / "models" / "mobilenetv2_tuned_v4_class_names.json")
))

IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

_model = None
_class_names = None


# ------------------------------------------------------------
# Load V4 model and matching labels
# ------------------------------------------------------------
def load_assets():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"V4 model not found: {MODEL_PATH}"
            )

        if not LABELS_PATH.is_file():
            raise FileNotFoundError(
                f"V4 class labels not found: {LABELS_PATH}"
            )

        model = keras.models.load_model(MODEL_PATH)

        with LABELS_PATH.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        if not isinstance(labels, list) or not labels:
            raise ValueError(
                "V4 class labels must be a non-empty JSON list."
            )

        if model.output_shape[-1] != len(labels):
            raise ValueError(
                f"Model outputs {model.output_shape[-1]} classes, "
                f"but the label file contains {len(labels)}."
            )

        if tuple(model.input_shape[1:3]) != IMAGE_SIZE:
            raise ValueError(
                f"Expected model input size {IMAGE_SIZE}, "
                f"but received {model.input_shape}."
            )

        _model = model
        _class_names = labels

    return _model, _class_names


# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------
@app.get("/api/health")
def health():
    try:
        model, labels = load_assets()

        return jsonify({
            "status": "ready",
            "model": MODEL_PATH.name,
            "classes": len(labels),
            "image_size": list(IMAGE_SIZE)
        })

    except Exception:
        app.logger.exception("Health check failed")
        return jsonify({
            "status": "error",
            "error": "The TomatoDoc model is not ready."
        }), 503


# ------------------------------------------------------------
# Disease prediction
# ------------------------------------------------------------
@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload a leaf image."
        }), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({
            "error": "No image was selected."
        }), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        image = image.resize(IMAGE_SIZE)

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "The image could not be read. Try a JPG, PNG, or WEBP file."
        }), 400

    try:
        model, labels = load_assets()

        # V4 includes MobileNetV2 preprocessing inside the model.
        # Pass RGB pixels in the 0-255 range; do not preprocess again.
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        predictions = np.asarray(
            model.predict(array, verbose=0)
        )[0].astype(float)

        if not np.all(np.isfinite(predictions)):
            raise ValueError("The model returned invalid prediction values.")

        if len(predictions) != len(labels):
            raise ValueError("The prediction and label counts do not match.")

        best_indices = np.argsort(predictions)[::-1][:3]
        best = int(best_indices[0])

        return jsonify({
            "prediction": labels[best],
            "confidence": round(
                float(predictions[best]) * 100, 2
            ),
            "top3": [
                {
                    "class_name": labels[int(i)],
                    "confidence": round(
                        float(predictions[int(i)]) * 100, 2
                    )
                }
                for i in best_indices
            ]
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


# ------------------------------------------------------------
# Upload size error
# ------------------------------------------------------------
@app.errorhandler(413)
def too_large(_error):
    return jsonify({
        "error": "Image is too large. Please use an image under 12 MB."
    }), 413


if __name__ == "__main__":
    # Local development only. Use a production WSGI server when deployed.
    app.run(host="127.0.0.1", port=5000, debug=False)
