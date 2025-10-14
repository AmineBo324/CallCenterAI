from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# ============================================
# CONFIGURATION
# ============================================

app = FastAPI(
    title="Transformer Ticket Classification API",
    description="Classification de tickets avec DistilBERT multilingue",
    version="1.0.0"
)

# Chemin du modèle - ADAPTER SELON TON CHEMIN
MODEL_DIR = r"C:\Users\OrdiOne\Desktop\MLops\models\fine_tuned_model"

# ============================================
# MÉTRIQUES PROMETHEUS
# ============================================

request_counter = Counter(
    'transformer_requests_total',
    'Nombre total de requêtes',
    ['status']
)

prediction_time = Histogram(
    'transformer_prediction_seconds',
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
                "text": "Bonjour, mon ordinateur portable ne démarre plus"
            }
        }

class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    processing_time: float

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================

tokenizer = None
model = None
label_encoder = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    
    # Mode évaluation
    model.eval()
    
    # GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ Modèle Transformer chargé depuis {MODEL_DIR}")
    print(f"✅ Device: {device}")
    print(f"✅ Classes: {label_encoder.classes_}")
    
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Point d'entrée de l'API"""
    return {
        "service": "Transformer Classification Service",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(model.device) if model else None
    }

@app.get("/health")
def health():
    """Vérification santé du service"""
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(model.device)}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Prédire la catégorie d'un ticket
    
    Args:
        request: Texte du ticket
        
    Returns:
        Label, confiance et scores pour chaque classe
    """
    start_time = time.time()
    
    try:
        # Validation
        if not request.text or len(request.text.strip()) == 0:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        
        if model is None or tokenizer is None or label_encoder is None:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # Prédiction
        with prediction_time.time():
            # Tokenization
            inputs = tokenizer(
                request.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Déplacer sur GPU si disponible
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Inférence
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Calcul des scores
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][predicted_class_id].item()
        
        # Décoder le label
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
        
        # Créer dictionnaire des scores
        scores_dict = {
            label_encoder.inverse_transform([i])[0]: float(probs[0][i])
            for i in range(len(label_encoder.classes_))
        }
        
        # Trier par score décroissant
        scores_dict = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))
        
        processing_time = time.time() - start_time
        
        request_counter.labels(status='success').inc()
        
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
    uvicorn.run(app, host="0.0.0.0", port=8002)