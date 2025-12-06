"""
Service API FastAPI pour le modèle Transformer (DistilBERT)
+ intégration Prometheus et MLflow (uniforme avec TF-IDF)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import mlflow
import mlflow.pytorch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Transformer Ticket Classification API",
    description="Classification de tickets avec DistilBERT + MLflow Tracking",
    version="1.2.0"
)

# CORS (développement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PATHS ET CONFIGURATION
# ============================================

MODEL_DIR = "./models/fine_tuned_model"

# ============================================
#  MLflow Configuration
# ============================================

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Transformer_Classifier")

# ============================================
#  PROMETHEUS - METRICS UNIFIÉES
# ============================================

# 🔥 SAME METRICS AS TF-IDF SERVICE
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

# Specific transformer metrics
request_counter = Counter(
    'transformer_requests_total',
    'Nombre total de requêtes Transformer',
    ['status']
)

prediction_time = Histogram(
    'transformer_prediction_seconds',
    'Temps de prédiction Transformer'
)

# ============================================
#  MODELS
# ============================================

class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Bonjour, mon ordinateur portable ne démarre plus"
            }
        }

class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    processing_time: float

# ============================================
# CHARGEMENT DU MODELE
# ============================================

tokenizer = None
model = None
label_encoder = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("✅ Modèle Transformer chargé")

except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "service": "Transformer Classification Service",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(model.device) if model else None
    }

@app.get("/health")
def health():
    status = {
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "label_encoder_loaded": label_encoder is not None
    }
    if not all(status.values()):
        raise HTTPException(status_code=503, detail=status)
    return {"status": "healthy", "device": str(model.device)}

# ============================================
#  PREDICT
# ============================================

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    try:
        if not request.text.strip():
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")

        if model is None:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=503, detail="Modèle non chargé")

        # ---------------------------
        # 🔥 Prediction
        # ---------------------------
        with prediction_time.time():
            inputs = tokenizer(
                request.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        class_id = torch.argmax(probs).item()
        confidence = float(probs[class_id])
        predicted_label = label_encoder.classes_[class_id]

        scores_dict = {
            label_encoder.classes_[i]: float(probs[i])
            for i in range(len(label_encoder.classes_))
        }
        scores_dict = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))

        processing_time = time.time() - start_time
        request_counter.labels(status='success').inc()

        # ============================================
        # ⭐ PROMETHEUS - SAME AS TF-IDF ⭐
        # ============================================
        MODEL_REQUESTS.labels("transformer").inc()
        LABEL_COUNTER.labels(label=predicted_label).inc()
        CONFIDENCE_HIST.observe(confidence)
        PROCESSING_TIME.observe(processing_time)

        # ============================================
        # MLflow
        # ============================================
        with mlflow.start_run(run_name="transformer_prediction"):
            mlflow.log_param("model_type", "DistilBERT")
            mlflow.log_param("device", str(model.device))
            mlflow.log_param("text_length", len(request.text))
            mlflow.log_param("predicted_label", predicted_label)

            short_text = request.text[:150] + "..." if len(request.text) > 150 else request.text
            mlflow.log_param("input_text", short_text)

            top3 = list(scores_dict.items())[:3]
            mlflow.log_param("top3_predictions", str(top3))

            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("processing_time", processing_time)
            mlflow.log_metric("num_classes", len(label_encoder.classes_))

        return PredictResponse(
            label=predicted_label,
            confidence=confidence,
            scores=scores_dict,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        request_counter.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================
# LANCEMENT
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
