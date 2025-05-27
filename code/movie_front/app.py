from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Créer le dossier d'upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# URL vers les endpoints de l'API backend (conteneur "movie_api")
PREDICT_API_URL = "http://movie_api:5000/predict"
RECOMMEND_API_URL = "http://movie_api:5000/recommend_poster"
RECOMMEND_PLOT_API_URL = "http://movie_api:5000/recommend_plot_movie"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier trouvé"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Fichier non valide"}), 400
        
        # Sauvegarde temporaire
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        with open(filepath, "rb") as img:
            response = requests.post(PREDICT_API_URL, data=img)

        if response.status_code == 200:
            prediction = response.json().get("prediction", "Erreur")
        else:
            prediction = "Erreur de prédiction"

        return jsonify({"prediction": prediction})

    return render_template("index.html")


@app.route("/recommend_poster", methods=["POST"])
def recommend():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400

    # Sauvegarde temporaire
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    with open(filepath, "rb") as img:
        response = requests.post(RECOMMEND_API_URL, data=img)

    if response.status_code == 200:
        result = response.json()
        recommendations = result.get("recommendations", [])
    else:
        recommendations = []

    return jsonify({"recommendations": recommendations})

@app.route('/MLP-20M/<path:filename>')
def serve_mlp20m(filename):
    return send_from_directory('MLP-20M', filename)



@app.route("/recommend_plot", methods=["POST"])
def recommend_plot():
    data = request.get_json()

    # Récupération de la description et de la méthode
    description = data.get("description", "")
    method = data.get("method", "Bert")  # Valeur par défaut = BERT

    if not description:
        return jsonify({"error": "Aucune description fournie"}), 400

    payload = {
        "description": description,
        "method": method
    }

    try:
        response = requests.post(RECOMMEND_PLOT_API_URL, json=payload)
        if response.status_code == 200:
            results = response.json().get("recommendations", [])
        else:
            return jsonify({"error": "Erreur lors de l'appel à l'API"}), 500
    except Exception as e:
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

    return jsonify({"recommendations": results})
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
