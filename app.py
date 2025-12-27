from flask import Flask, request, render_template, url_for, jsonify
import os
import uuid
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
TMP_FOLDER = "tmp_parts"
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL = None
CLASS_NAMES = []

def load_model_and_classes():
    global MODEL, CLASS_NAMES
    try:
        if os.path.exists("maize_deficiency_model_final.keras"):
            MODEL = load_model("maize_deficiency_model_final.keras")
            print("Model loaded successfully")
        else:
            print("Warning: maize_deficiency_model_final.keras not found")
        
        if os.path.exists("maize_deficiency_classes.txt"):
            with open("maize_deficiency_classes.txt", "r") as f:
                CLASS_NAMES = [line.strip() for line in f.readlines()]
            print(f"Classes loaded: {CLASS_NAMES}")
    except Exception as e:
        print(f"Error loading model: {e}")

load_model_and_classes()

def is_blurry(image_path, threshold=300.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    print(f"Laplacian variance: {laplacian_var}")
    return laplacian_var > threshold

def save_bytes_and_url(data_bytes, ext="jpg"):
    fname = f"{uuid.uuid4()}.{ext}"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    with open(fpath, "wb") as f:
        f.write(data_bytes)
    return url_for("static", filename=f"uploads/{fname}", _external=True), fpath

def predict_deficiency(img_path):
    if MODEL is None or not CLASS_NAMES:
        return None, 0.0, "Model not loaded"
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        predictions = MODEL.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, 0.0, str(e)

def get_remedies(deficiency_class):
    remedies_db = {
        "HEALTHY": {
            "analysis": "Crop appears healthy with no nutrient deficiency detected",
            "remedies": [
                "Maintain current fertilization schedule",
                "Continue regular soil moisture monitoring",
                "Practice crop rotation in next season"
            ]
        },
        "NITROGEN": {
            "analysis": "Nitrogen deficiency causes yellowing of older leaves and stunted growth",
            "remedies": [
                "Apply compost or well-rotted manure (5-10 tons per hectare)",
                "Use green manure crops like legumes in rotation",
                "Apply vermicompost (2-3 tons per hectare)",
                "Plant nitrogen-fixing cover crops between seasons",
                "Use organic mulch to improve soil nitrogen retention"
            ]
        },
        "PHOSPHOROUS": {
            "analysis": "Phosphorus deficiency shows purple or dark green leaves and slow growth",
            "remedies": [
                "Apply rock phosphate (200-400 kg per hectare)",
                "Use bone meal as organic phosphorus source",
                "Add composted animal manure rich in phosphorus",
                "Incorporate wood ash into soil (moderate amounts)",
                "Maintain soil pH between 6.0-7.0 for optimal phosphorus availability"
            ]
        },
        "POTASSIUM": {
            "analysis": "Potassium deficiency results in scorched leaf edges and weak stalks",
            "remedies": [
                "Apply wood ash (1-2 tons per hectare)",
                "Use kelp meal or seaweed extracts",
                "Add greensand (natural potassium source)",
                "Incorporate banana peels or plant residues into compost",
                "Apply composted farmyard manure regularly"
            ]
        },
        "ZINC": {
            "analysis": "Zinc deficiency causes white or yellow bands on leaves and delayed maturity",
            "remedies": [
                "Apply zinc sulfate through soil (10-25 kg per hectare)",
                "Foliar spray of zinc solution during vegetative stage",
                "Add organic matter to improve zinc availability",
                "Avoid over-liming soil which reduces zinc uptake",
                "Use zinc-enriched compost"
            ]
        }
    }
    
    if deficiency_class in remedies_db:
        return remedies_db[deficiency_class]
    else:
        return {
            "analysis": "Deficiency type identified but specific treatment not available",
            "remedies": [
                "Consult agricultural extension office",
                "Get soil tested for nutrient levels",
                "Apply balanced organic fertilizer"
            ]
        }

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        ext = file.filename.rsplit(".", 1)[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_url = url_for("static", filename=f"uploads/{filename}")
        return render_template("upload.html", img_url=img_url, filename=filename)

    return render_template("upload.html", img_url=None)

@app.route("/api/upload", methods=["POST"])
def api_upload():
    print("Received request to /api/upload")
    print("Content-Type:", request.headers.get("Content-Type"))
    print("Content-Length:", request.content_length)

    file = request.files.get("file") or request.files.get("image")
    if not file:
        files_list = list(request.files.values())
        file = files_list[0] if files_list else None

    if not file:
        return jsonify({"error": "No file received"}), 400

    ext = file.filename.rsplit(".", 1)[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img_url = url_for("static", filename=f"uploads/{filename}", _external=True)
    return jsonify({"img_url": img_url}), 200

@app.route("/api/upload_b64_chunk", methods=["POST"])
def upload_b64_chunk():
    upload_id = request.args.get("id", None)
    try:
        idx = int(request.args.get("idx", "0"))
        total = int(request.args.get("total", "0"))
    except ValueError:
        return jsonify({"error": "Invalid idx/total"}), 400

    if not upload_id or idx <= 0 or total <= 0:
        return jsonify({"error": "Missing id/idx/total"}), 400

    chunk_text = request.get_data(as_text=True)
    if chunk_text is None:
        return jsonify({"error": "Empty chunk"}), 400

    part_path = os.path.join(TMP_FOLDER, f"{upload_id}.part")
    try:
        with open(part_path, "a", encoding="utf-8") as f:
            f.write(chunk_text)
    except Exception as e:
        return jsonify({"error": "Failed to save chunk", "err": str(e)}), 500

    if idx == total:
        try:
            with open(part_path, "r", encoding="utf-8") as f:
                whole_b64 = f.read()
            whole_b64 = "".join(whole_b64.split())
            data = base64.b64decode(whole_b64)
        except Exception as e:
            try:
                os.remove(part_path)
            except:
                pass
            return jsonify({"error": "Base64 decode failed", "err": str(e)}), 400

        url, saved_path = save_bytes_and_url(data, ext="jpg")
        try:
            os.remove(part_path)
        except:
            pass
        return jsonify({"img_url": url, "saved": saved_path}), 200

    return jsonify({"status": "ok", "received": idx, "total": total}), 200

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 400
    
    if is_blurry(filepath):
        return jsonify({"error": "Image is too blurry, please retake"}), 400
    
    predicted_class, confidence, error = predict_deficiency(filepath)
    
    if error:
        return jsonify({"error": f"Prediction failed: {error}"}), 500
    
    remedies_info = get_remedies(predicted_class)

    result = {
        "class": predicted_class,
        "confidence": round(confidence, 4),
        "analysis": remedies_info["analysis"],
        "remedies": remedies_info["remedies"]
    }

    return jsonify(result), 200

@app.route("/predict", methods=["POST"])
def predict():
    filename = request.form.get("filename")
    
    if not filename:
        return "Missing filename", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    img_url = url_for("static", filename=f"uploads/{filename}")
    
    if not os.path.exists(filepath):
        return "File not found", 400

    if is_blurry(filepath):
        return render_template("upload.html", 
                             img_url=img_url, 
                             filename=filename,
                             error="Image is too blurry, please reupload")
    
    predicted_class, confidence, error = predict_deficiency(filepath)
    
    if error:
        return render_template("upload.html", 
                             img_url=img_url, 
                             filename=filename,
                             error=f"Prediction failed: {error}")
    
    remedies_info = get_remedies(predicted_class)

    result = {
        "class": predicted_class,
        "confidence": round(confidence, 4),
        "analysis": remedies_info["analysis"],
        "remedies": remedies_info["remedies"]
    }

    print("Confidence: ", round(confidence, 4)*100)

    return render_template("upload.html", 
                         img_url=img_url, 
                         filename=filename,
                         results=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)