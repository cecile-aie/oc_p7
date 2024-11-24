import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
import mlflow.pyfunc
from deep_translator import GoogleTranslator
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from datetime import datetime

# Récupérer la chaîne de connexion depuis les variables d'environnement
CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_TWEETALERT_CONNECTION_STRING", "")
if not CONNECTION_STRING:
    raise EnvironmentError(
        "La variable d'environnement APPLICATIONINSIGHTS_TWEETALERT_CONNECTION_STRING est manquante."
    )

# Configurer OpenTelemetry avec un TracerProvider spécifique
manual_provider = TracerProvider()
exporter = AzureMonitorTraceExporter(connection_string=CONNECTION_STRING)
manual_provider.add_span_processor(BatchSpanProcessor(exporter))
manual_tracer = trace.get_tracer_provider().get_tracer(__name__)

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
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        print(f"Erreur lors de la traduction : {e}")
        return text

# Page d'accueil avec un formulaire pour soumettre des phrases
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text_input = request.form.get("text")
        feedback = request.form.get("feedback")

        MAX_TEXT_LENGTH = 500
        if text_input and len(text_input) > MAX_TEXT_LENGTH:
            return render_template(
                "index.html",
                error="Le texte est trop long. Veuillez entrer un texte de moins de 500 caractères.",
                sentiment=None,
                input_text=None,
                translated_text=None,
                feedback_received=False,
            )

        if text_input:
            try:
                translated_text = translate_to_english(text_input)

                input_data = pd.DataFrame([{"text": translated_text}])
                predictions = model.predict(input_data)
                sentiment = "Positif" if predictions[0] == 0 else "Négatif"

                if feedback == "incorrect":
                    log_incorrect_prediction(translated_text, sentiment)
                    return render_template(
                        "index.html",
                        sentiment=None,
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

        translated_text = translate_to_english(original_text)

        input_data = pd.DataFrame([{"text": translated_text}])
        predictions = model.predict(input_data)

        return jsonify(
            {
                "original_text": original_text,
                "translated_text": translated_text,
                "prediction": predictions.tolist(),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def log_incorrect_prediction(text, sentiment):
    """Enregistre les prédictions incorrectes dans Azure Application Insights."""
    with manual_tracer.start_as_current_span("IncorrectPrediction") as span:
        span.set_attribute("text", text)
        span.set_attribute("predicted_sentiment", sentiment)
        span.set_attribute("timestamp", datetime.utcnow().isoformat())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
