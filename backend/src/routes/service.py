from flask import Blueprint, request, jsonify
from typing import TypedDict
import numpy as np

from ..models.scripts.dataset import LetterDataset

from ..services.converter import matrix2image
from ..services.generator import Generate, Predict
from ..utils import idx_to_letter

from ..models.scripts.NN import LetterRecognitionModel
from torch import load

# create a blueprint
bp = Blueprint("service", __name__)


class ImageData(TypedDict): ...


class MatrixData(TypedDict): ...


FONT = {
    0: "ArialCEBoldItalic",
    1: "Roboto-Italic-VariableFont_wdth,wght",
    2: "georgiaz",
    3: "Helvetica",
    4: "Lato-Bold",
    5: "OpenSans-VariableFont_wdth,wght",
    6: "CourierPrime-Italic",
    7: "times",
    8: "NotoSans-Black",
    9: "arialceb",
}


@bp.route("/service/predict", methods=["POST"])
def predict():
    data = request.json
    if data is None:
        return jsonify({"error": "No data provided"})
    image = data["matrix"]

    # to ndarray
    image = np.array(image, dtype=np.uint8)

    # Initialize model
    model = LetterRecognitionModel(len(idx_to_letter))
    # import weights
    model.load_state_dict(  # type: ignore
        load(
            "./src/models/trained_models/best_letter_recognition_model.pt",
            weights_only=True,
        )
    )

    # set to evaluation mode
    model.eval()  # type: ignore
    model = model.cpu()  # type: ignore

    letter = Predict(model, image)
    font = FONT[0]

    return jsonify({"letter": letter, "font": font}), 200


@bp.route("/service/generate", methods=["POST"])
def generate():
    data = request.json
    if data is None:
        return jsonify({"error": "No data provided"}), 400

    # check if the data has {letter, font_idx, matrix}
    if "letter" not in data or "font_idx" not in data or "matrix" not in data:
        return jsonify({"error": "Invalid data"}), 400

    matrix = data["matrix"]
    letter = data["letter"]
    font_idx = data["font_idx"]

    # to ndarray
    matrix = np.array(matrix, dtype=np.uint8)

    matrix = Generate(matrix, letter, FONT[font_idx])

    try:
        image = matrix2image(matrix)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

    return jsonify({"image": image}), 200


@bp.route("/service/fonts", methods=["GET"])
def convert_image_to_matrix():
    return jsonify({"fonts": FONT}), 200
