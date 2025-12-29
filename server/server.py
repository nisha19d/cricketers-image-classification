from flask import Flask, request, jsonify, render_template
import util
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

util.load_saved_artifacts()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/classify_image", methods=["POST"])
def classify_image():
    image_data = request.json["image"]
    player, probabilities = util.classify_image(image_data)

    return jsonify({
        "player": player,
        "probabilities": probabilities
    })


if __name__ == "__main__":
    app.run(port=5050, debug=True)
