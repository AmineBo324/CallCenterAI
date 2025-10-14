"""
Service API FastAPI pour le modèle TF-IDF + SVM
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib
import time

# ============================================
# CONFIGURATION
# ============================================

app = FastAPI(
    title="TF-IDF Ticket Classification API",
    description="Classification de tickets avec TF-IDF + SVM",
    version="1.0.0"
)

# Chemin du modèle - ADAPTER SELON TON CHEMIN
MODEL_PATH = r"C:\Users\OrdiOne\Desktop\MLops\models\ticket_classifier_model.pkl"

# ============================================
# MÉTRIQUES PROMETHEUS
# ============================================

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
    """
    Prédire la catégorie d'un ticket
    
    Args:
        request: Texte du ticket
        
    Returns:
        Label, confiance et probabilités pour chaque classe
    """
    start_time = time.time()
    
    try:
        # Validation
        if not request.text or len(request.text.strip()) == 0:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        
        if model is None:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # Prédiction
        with prediction_time.time():
            prediction = model.predict([request.text])[0]
            probabilities = model.predict_proba([request.text])[0]
        
        # Confiance (probabilité max)
        confidence = float(max(probabilities))
        
        # Créer dictionnaire des probabilités
        proba_dict = {
            cls: float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }
        
        # Trier par probabilité décroissante
        proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
        
        processing_time = time.time() - start_time
        
        request_counter.labels(status='success').inc()
        
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