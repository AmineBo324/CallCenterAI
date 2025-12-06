from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib
import time
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="TF-IDF Ticket Classification API",
    description="Classification de tickets avec TF-IDF + SVM + MLflow Tracking",
    version="1.1.0"
)
# ✅ Autoriser les requêtes depuis n’importe quelle origine (pour développement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu peux remplacer "*" par ["http://127.0.0.1:5500"] par ex.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./models/ticket_classifier_model.pkl"

# ============================================
# CONFIG MLflow
# ============================================

# 🔗 Adresse du serveur MLflow (modifie selon ton setup Docker)
mlflow.set_tracking_uri("http://mlflow:5000")

# 🔧 Nom de l'expérience (créée automatiquement si elle n'existe pas)
mlflow.set_experiment("TFIDF_Classifier")

# ============================================
# MÉTRIQUES PROMETHEUS
# ============================================

MODEL_REQUESTS = Counter(
    'ml_model_requests_total',
    'Total predictions per model',
    ['model_type']
)

LABEL_COUNTER = Counter(
    'ml_predictions_labels_total',
    'Total predictions per label',
    ['label']
)

CONFIDENCE_HIST = Histogram(
    'ml_prediction_confidence',
    'Prediction confidence distribution'
)

PROCESSING_TIME = Histogram(
    'ml_prediction_latency_seconds',
    'Latency of predictions'
)



request_counter = Counter(
    'tfidf_requests_total',
    'Nombre total de requêtes',
    ['status']
)

prediction_time = Histogram(
    'tfidf_prediction_seconds',
    'Temps de prédiction'
)

# ============================================
# MODÈLES PYDANTIC
# ============================================

class PredictRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "my laptop is broken and won't turn on"
            }
        }

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    processing_time: float

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle TF-IDF chargé depuis {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Point d'entrée de l'API"""
    return {
        "service": "TF-IDF Classification Service",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    """Vérification santé du service"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()
    
    try:
        if not request.text or len(request.text.strip()) == 0:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        
        if model is None:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # --- PRÉDICTION ---
        with prediction_time.time():
            prediction = model.predict([request.text])[0]
            probabilities = model.predict_proba([request.text])[0]
        
        confidence = float(max(probabilities))
        proba_dict = {
            cls: float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }
        proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
        
        processing_time = time.time() - start_time
        request_counter.labels(status='success').inc()

        # ==========================================
        # ⭐⭐⭐ PROMETHEUS METRICS (add them HERE) ⭐⭐⭐
        # ==========================================

        MODEL_REQUESTS.labels("tfidf").inc()
        LABEL_COUNTER.labels(label=prediction).inc()
        CONFIDENCE_HIST.observe(confidence)
        PROCESSING_TIME.observe(processing_time)

        # ==========================================

        # --- LOG MLflow ---
        with mlflow.start_run(run_name="tfidf_prediction"):
            mlflow.log_param("model_type", "TF-IDF + SVM")
            mlflow.log_param("text_length", len(request.text))
            mlflow.log_param("predicted_label", prediction)

            short_text = request.text[:150] + "..." if len(request.text) > 150 else request.text
            mlflow.log_param("input_text", short_text)

            top3 = list(proba_dict.items())[:3]
            mlflow.log_param("top3_predictions", str(top3))

            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("processing_time", processing_time)
            mlflow.log_metric("num_classes", len(model.classes_))

        return PredictResponse(
            label=prediction,
            confidence=confidence,
            probabilities=proba_dict,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        request_counter.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Endpoint pour Prometheus"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ============================================
# LANCEMENT
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
