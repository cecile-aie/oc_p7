import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
import mlflow.pyfunc
from deep_translator import GoogleTranslator
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
from datetime import datetime

# Configuration du logger pour Application Insights
instrumentation_key = "4071129e-e96b-494a-b0c7-24e6dac41b18"  
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={instrumentation_key}"))
logger.setLevel(logging.INFO)


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

# Fonction pour traduire le texte en anglais
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        # Si une erreur survient, retourner le texte original
        print(f"Erreur lors de la traduction : {e}")
        return text

def log_incorrect_prediction(translated_text, sentiment):
    logger.info(
        "Prédiction incorrecte signalée",
        extra={
            "custom_dimensions": {  
                "translated_text": translated_text,
                "sentiment": sentiment
            }
        }
    )

# Page d'accueil avec un formulaire pour soumettre des phrases
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        feedback = request.form.get("feedback")

         # Vérifier si le texte dépasse la limite de taille
        MAX_TEXT_LENGTH = 500  
        if text_input and len(text_input) > MAX_TEXT_LENGTH:
                        return render_template(
                "index.html",
                error="Le texte est trop long. Veuillez entrer un texte de moins de 500 caractères.",
                sentiment=None,
                input_text=None,
                translated_text=None,
                feedback_received=False
            )
        
        if text_input:
            try:
                # Traduire le texte en anglais
                translated_text = translate_to_english(text_input)
                
                # Effectuer la prédiction avec le modèle
                input_data = pd.DataFrame([{"text": translated_text}])
                predictions = model.predict(input_data)
                sentiment = "Positif" if predictions[0] == 0 else "Négatif"
                
                # Si l'utilisateur donne un feedback négatif, remonter l'information
                if feedback == "incorrect":
                    log_incorrect_prediction(translated_text, sentiment)
                    return render_template(
                        "index.html",
                        sentiment=None,  # Réinitialise les valeurs affichées
                        input_text=None,
                        translated_text=None,
                        feedback_received=True,
                    )
                
                return render_template(
                    "index.html", 
                    sentiment=sentiment, 
                    input_text=text_input, 
                    translated_text=translated_text,
                    feedback_received=True,
                )
            except Exception as e:
                logger.error(f"Erreur : {e}")
                return render_template("index.html", error=f"Erreur : {str(e)}")
        else:
            return render_template("index.html", error="Veuillez entrer une phrase.")
    return render_template("index.html")

# Définir l'endpoint pour faire des prédictions via API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        original_text = data.get("text", "")
        
        # Traduire le texte en anglais
        translated_text = translate_to_english(original_text)
        
        # Effectuer la prédiction
        input_data = pd.DataFrame([{"text": translated_text}])
        predictions = model.predict(input_data)
        
        return jsonify({
            "original_text": original_text,
            "translated_text": translated_text,
            "prediction": predictions.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Détecter le port pour Azure Web App, sinon utiliser 5001
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)