from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import json
from price_service import search_products

# OCR imports
import pytesseract
from PIL import Image

# --------------------
# CONFIG
# --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "product_model.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------
# LOAD CNN MODEL
# --------------------
print("Loading model...")
model = load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_dict = json.load(f)

CLASS_NAMES = [None] * (max(class_dict.values()) + 1)
for name, idx in class_dict.items():
    CLASS_NAMES[idx] = name

print("Model loaded successfully")

# --------------------
# OCR BRAND KEYWORDS
# --------------------
BRAND_KEYWORDS = {
    "samsung": "Samsung",
    "galaxy": "Samsung",
    "apple": "Apple",
    "iphone": "Apple",
    "realme": "Realme",
    "redmi": "Redmi",
    "oppo": "Oppo",
    "oneplus": "OnePlus",
    "vivo": "Vivo"
}

# --------------------
# FLASK APP
# --------------------
app = Flask(__name__)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --------------------
# OCR FIRST (RESTORED)
# --------------------
def ocr_detect_brand(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(
            img,
            config="--oem 3 --psm 6",
            lang="eng"
        ).lower()

        print("OCR TEXT:", text[:300])

        for key, brand in BRAND_KEYWORDS.items():
            if key in text:
                print("OCR BRAND FOUND:", brand)
                return brand

        return None

    except Exception as e:
        print("OCR ERROR:", e)
        return None

# --------------------
# CNN FALLBACK
# --------------------
def cnn_predict_brand(img_path):
    img = load_img(img_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    return CLASS_NAMES[idx], conf

# --------------------
# OCR → CNN DECISION LOGIC
# --------------------
def predict_brand(img_path):
    # 1️⃣ OCR FIRST
    brand = ocr_detect_brand(img_path)
    if brand:
        return brand, "OCR"

    # 2️⃣ CNN FALLBACK
    brand, conf = cnn_predict_brand(img_path)
    return brand, f"CNN"

# --------------------
# MAIN ROUTE
# --------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    label = None
    source = None
    image_url = None
    prices = []
    best_store = None
    model_name = None

    # ---------- STEP 1: IMAGE UPLOAD ----------
    if request.method == "POST" and "image" in request.files:
        file = request.files.get("image")
        if file and file.filename:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            label, source = predict_brand(file_path)
            image_url = f"/uploads/{file.filename}"

    # ---------- STEP 2: MODEL TEXT SUBMISSION ----------
    if request.method == "POST" and "model_name" in request.form:
        label = request.form.get("brand")
        model_name = request.form.get("model_name")
        image_url = request.form.get("image_url")
        if label and model_name:
            query = f"{label} {model_name} smartphone Amazon Flipkart Croma"
            prices, best_store = search_products(query)

    return render_template(
        "index.html",
        label=label,
        source=source,
        image_url=image_url,
        prices=prices,
        best_store=best_store
    )

# --------------------
# RUN SERVER
# --------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
