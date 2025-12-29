import json
import numpy as np
import cv2
import base64
import os
import joblib

from wavelet import w2d

__class_name_to_number = {}
__model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __class_name_to_number
    global __model

    class_dict_path = os.path.join(ARTIFACTS_DIR, "class_dictionary.json")
    model_path = os.path.join(ARTIFACTS_DIR, "saved_model.pkl")

    with open(class_dict_path, "r") as f:
        __class_name_to_number = json.load(f)

    __model = joblib.load(model_path)

    print("loading saved artifacts...done")


def get_cv2_image_from_base64(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)


def classify_image(base64_img):
    if __model is None:
        raise Exception("Model not loaded")

    img = get_cv2_image_from_base64(base64_img)

    if img is None:
        raise Exception("Invalid image")

    img = cv2.resize(img, (32, 32))

    img_har = w2d(img, 'db1', 5)
    img_har = cv2.resize(img_har, (32, 32))

    combined = np.vstack((
        img.reshape(32 * 32 * 3, 1),
        img_har.reshape(32 * 32, 1)
    ))

    final = combined.reshape(1, -1).astype(float)

    raw_scores = __model.decision_function(final)[0]
    probabilities = softmax(raw_scores)

    result = {}
    for name, index in __class_name_to_number.items():
        result[name] = round(probabilities[index] * 100, 2)

    predicted_player = max(result, key=result.get)

    print("Probability sum:", sum(result.values()))

    return predicted_player, result
