"""
Service Agent IA - Routage intelligent entre TF-IDF et Transformer
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import requests
import re
import time

# ============================================
# CONFIGURATION
# ============================================

app = FastAPI(
    title="AI Agent - Intelligent Router",
    description="Routage intelligent des tickets vers le bon modèle avec nettoyage PII",
    version="1.0.0"
)

# URLs des services
TFIDF_URL = "http://localhost:8001"
TRANSFORMER_URL = "http://localhost:8002"

# ============================================
# MÉTRIQUES PROMETHEUS
# ============================================

request_counter = Counter(
    'agent_requests_total',
    'Nombre total de requêtes',
    ['status']
)

routing_counter = Counter(
    'agent_routing_decisions',
    'Décisions de routage',
    ['model', 'reason']
)

# ============================================
# MODÈLES PYDANTIC
# ============================================

class PredictRequest(BaseModel):
    text: str
    force_model: str = None  # 'tfidf' ou 'transformer' pour forcer
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "my laptop screen is broken",
                "force_model": None
            }
        }

class AgentResponse(BaseModel):
    label: str
    confidence: float
    model_used: str
    routing_reason: str
    pii_detected: bool
    detected_language: str
    text_length: int
    processing_time: float
    explanation: str

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def scrub_pii(text: str) -> tuple[str, bool]:
    """
    Détecte et masque les informations personnelles (PII)
    
    Returns:
        (texte_nettoyé, pii_trouvé)
    """
    pii_detected = False
    cleaned_text = text
    
    # Patterns pour différents types de PII
    patterns = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b\d{10,}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'CREDIT_CARD': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'IP': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    for pii_type, pattern in patterns.items():
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            pii_detected = True
            cleaned_text = re.sub(pattern, f'[{pii_type}]', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text, pii_detected

def detect_language(text: str) -> str:
    """
    Détection simple de la langue
    
    Returns:
        Code langue: 'en', 'fr', 'ar'
    """
    # Caractères arabes
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    
    # Mots français communs
    french_words = ['bonjour', 'merci', 'problème', 'erreur', 'compte', 'mot', 'passe']
    text_lower = text.lower()
    if any(word in text_lower for word in french_words):
        return 'fr'
    
    return 'en'

def decide_routing(text: str, cleaned_text: str) -> tuple[str, str, str]:
    """
    Décide quel modèle utiliser
    
    Returns:
        (model_name, reason, explanation)
    """
    text_length = len(cleaned_text.split())
    language = detect_language(cleaned_text)
    
    # Règle 1: Texte très court 
    if text_length < 10:
        return (
            'tfidf',
            'short_text',
            f"Texte court ({text_length} mots) → TF-IDF pour une réponse rapide"
        )
    
    # Règle 2: Langue non-anglaise 
    if language != 'en':
        return (
            'transformer',
            f'multilingual_{language}',
            f"Langue détectée: {language} → Transformer multilingue pour meilleure précision"
        )
    
    # Règle 3: Texte long → Transformer 
    if text_length > 25:
        return (
            'transformer',
            'long_complex_text',
            f"Texte long ({text_length} mots) → Transformer pour analyse contextuelle"
        )
    
    # Règle 4: Texte moyen en anglais → TF-IDF 
    return (
        'tfidf',
        'standard_case',
        f"Texte standard en anglais ({text_length} mots) → TF-IDF (rapide et efficace)"
    )

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Point d'entrée de l'API"""
    return {
        "service": "AI Agent - Intelligent Router",
        "status": "running",
        "tfidf_service": TFIDF_URL,
        "transformer_service": TRANSFORMER_URL
    }

@app.get("/health")
def health():
    """Vérification santé du service et des backends"""
    health_status = {"agent": "healthy", "backends": {}}
    
    # Check TF-IDF
    try:
        r = requests.get(f"{TFIDF_URL}/health", timeout=2)
        health_status["backends"]["tfidf"] = "healthy" if r.status_code == 200 else "unhealthy"
    except:
        health_status["backends"]["tfidf"] = "unreachable"
    
    # Check Transformer
    try:
        r = requests.get(f"{TRANSFORMER_URL}/health", timeout=2)
        health_status["backends"]["transformer"] = "healthy" if r.status_code == 200 else "unhealthy"
    except:
        health_status["backends"]["transformer"] = "unreachable"
    
    return health_status

@app.get("/routing/rules")
def routing_rules():
    """Informations sur les règles de routage"""
    return {
        "rules": {
            "tfidf": [
                "Textes courts (< 10 mots)",
                "Textes moyens en anglais (10-50 mots)",
                "Cas standards nécessitant une réponse rapide"
            ],
            "transformer": [
                "Textes multilingues (FR, AR, etc.)",
                "Textes longs et complexes (> 25 mots)",
                "Cas nécessitant une analyse contextuelle approfondie"
            ]
        },
        "pii_detection": {
            "types": ["Email", "Phone", "SSN", "Credit Card", "IP Address"],
            "action": "Automatic masking before processing"
        }
    }

@app.post("/predict", response_model=AgentResponse)
def predict(request: PredictRequest):
    """
    Prédire la catégorie d'un ticket avec routage intelligent
    
    Le système:
    1. Nettoie les PII (données personnelles)
    2. Analyse le texte (langue, longueur)
    3. Route vers le modèle approprié
    4. Retourne la prédiction avec explication
    """
    start_time = time.time()
    
    try:
        # Validation
        if not request.text or len(request.text.strip()) == 0:
            request_counter.labels(status='error').inc()
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        
        # Étape 1: Nettoyage PII
        cleaned_text, pii_detected = scrub_pii(request.text)
        
        # Étape 2: Analyse du texte
        text_length = len(cleaned_text.split())
        detected_language = detect_language(cleaned_text)
        
        # Étape 3: Décision de routage
        if request.force_model:
            model_to_use = request.force_model.lower()
            routing_reason = 'forced_by_user'
            explanation = f"Modèle forcé par l'utilisateur: {model_to_use}"
        else:
            model_to_use, routing_reason, explanation = decide_routing(
                request.text, cleaned_text
            )
        
        routing_counter.labels(model=model_to_use, reason=routing_reason).inc()
        
        # Étape 4: Appel au service approprié
        service_url = TFIDF_URL if model_to_use == 'tfidf' else TRANSFORMER_URL
        
        try:
            response = requests.post(
                f"{service_url}/predict",
                json={"text": cleaned_text},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Service error: {response.status_code}")
            
            result = response.json()
            
        except requests.exceptions.RequestException as e:
            request_counter.labels(status='error').inc()
            raise HTTPException(
                status_code=503,
                detail=f"Backend service ({model_to_use}) unreachable: {str(e)}"
            )
        
        processing_time = time.time() - start_time
        
        request_counter.labels(status='success').inc()
        
        return AgentResponse(
            label=result['label'] if 'label' in result else result['predicted_category'],
            confidence=result['confidence'],
            model_used=model_to_use,
            routing_reason=routing_reason,
            pii_detected=pii_detected,
            detected_language=detected_language,
            text_length=text_length,
            processing_time=processing_time,
            explanation=explanation
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
    