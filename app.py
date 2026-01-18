"""
Flask demo app for AI WasteWise.

Routes:
- GET / : index page with upload form
- POST /predict : accepts image upload and returns predictions + guidance
"""
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
from src.model import load_model, predict_image

UPLOAD_FOLDER = "tmp_uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET", "dev-key")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (weights optional)
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "")
MODEL = load_model(MODEL_WEIGHTS)

GUIDANCE = {
    "plastic": "Place in Plastic bin if clean. Rinse bottles where possible.",
    "paper": "Place in Paper bin. Remove non-paper materials (plastic stickers).",
    "metal": "Place in Metal bin. Empty liquids and rinse if possible.",
    "glass": "Place in Glass bin. Avoid mixing with ceramics.",
    "organic": "Place in Organic/Compost bin. No plastics.",
    "ewaste": "Do not put in recycling bins. Hand over to e-waste collection."
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file provided")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        pil_image = Image.open(save_path)
        preds = predict_image(MODEL, pil_image, topk=3)
        # Map to guidance
        results = []
        for label, score in preds:
            results.append({
                "label": label,
                "score": round(score, 3),
                "guidance": GUIDANCE.get(label, "")
            })
        return render_template("results.html", results=results, filename=filename)
    else:
        flash("Unsupported file type")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)