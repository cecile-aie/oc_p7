import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
import mlflow.pyfunc

# Chemin vers le stockage monté
LOCAL_MODEL_PATH = "/mnt/azureblob"

# Charger le modèle MLflow depuis le chemin monté
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Le chemin {LOCAL_MODEL_PATH} n'existe pas ou le stockage n'est pas monté.")
    return mlflow.pyfunc.load_model(LOCAL_MODEL_PATH)

# Initialiser le modèle
model = load_model()

# Initialiser Flask
app = Flask(__name__)

# Page d'accueil avec un formulaire pour soumettre des phrases
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        if text_input:
            # Effectuer la prédiction avec le modèle
            input_data = pd.DataFrame([{"text": text_input}])
            predictions = model.predict(input_data)
            sentiment = "Positif" if predictions[0] == 0 else "Négatif"
            return render_template("index.html", sentiment=sentiment, input_text=text_input)
        else:
            return render_template("index.html", error="Veuillez entrer une phrase.")
    return render_template("index.html")

# Définir l'endpoint pour faire des prédictions via API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        predictions = model.predict(input_data)
        return jsonify({"prediction": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Détecter le port pour Azure Web App, sinon utiliser 5001
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
