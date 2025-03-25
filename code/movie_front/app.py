from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Assurez-vous que le dossier uploads existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_URL = "http://127.0.0.1:5000/predict"  # Adresse de ton API Flask de classification

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier trouvé"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Fichier non valide"}), 400
        
        # Sauvegarde temporaire du fichier
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Envoie de l’image à l’API Flask
        with open(filepath, "rb") as img:
            response = requests.post(API_URL, data=img)

        if response.status_code == 200:
            prediction = response.json().get("prediction", "Erreur")
        else:
            prediction = "Erreur de prédiction"

        return jsonify({"prediction": prediction, "image_url": filepath})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)
