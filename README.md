# CallCenterAI - Système de Classification Intelligent de Tickets

## 🎯 Vue d'ensemble

CallCenterAI est un système MLOps complet qui classifie automatiquement les tickets de support en utilisant des modèles d'IA avancés.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │───▶│   Agent IA       │───▶│   TF-IDF Model  │
│   (Port 3000)   │    │   (Port 8002)    │    │   (Port 8000)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │ Transformer     │
                       │ Model           │
                       │ (Port 8001)     │
                       └─────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prometheus    │    │    Grafana       │    │     MLflow      │
│   (Port 9090)   │    │   (Port 3000)    │    │   (Port 5000)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🤖 Modèles d'IA

### TF-IDF + SVM
- **Utilisation** : Textes courts, rapides
- **Accuracy** : 85.55%
- **Catégories** : Hardware, Access, Network, Software

### DistilBERT Transformer  
- **Utilisation** : Textes longs, multilingues
- **Accuracy** : 99.72%
- **Support** : Français, Anglais

### Agent Intelligent
- **Routage automatique** basé sur :
  - Longueur du texte
  - Langue détectée  
  - Complexité du contenu

## 🚀 Installation & Démarrage

### Prérequis
```bash
- Docker Desktop
- Python 3.11+
- Git
```

### Démarrage rapide
```bash
# 1. Cloner le projet
git clone <repo-url>
cd callcenterai

# 2. Lancer tous les services
docker compose up -d

# 3. Tester les services
python test_services.py
```

## 🔌 API Endpoints

### Agent IA (Port 8002)
```http
POST /predict
Content-Type: application/json

{
  "text": "Mon laptop ne fonctionne plus",
  "force_model": "tfidf" // optionnel
}

Response:
{
  "category": "Hardware",
  "confidence": 0.9367,
  "model_used": "tfidf",
  "routing_reason": "Langue fr détectée → Transformer multilingue",
  "text_length": 5,
  "detected_language": "fr"
}
```

### TF-IDF Service (Port 8000)
```http
POST /predict
{
  "text": "password reset"
}

Response:
{
  "category": "Access",
  "confidence": 0.9333,
  "model": "TF-IDF + SVM"
}
```

### Transformer Service (Port 8001)
```http
POST /predict
{
  "text": "My computer screen is broken and I need urgent help"
}

Response:
{
  "category": "Hardware", 
  "confidence": 0.9972,
  "model": "DistilBERT-multilingual"
}
```

## 📊 Monitoring & MLOps

### Interfaces Web
- **Grafana** : http://localhost:3000 (admin/admin)
- **Prometheus** : http://localhost:9090  
- **MLflow** : http://localhost:5000

### Métriques Trackées
- Accuracy par modèle
- Temps de réponse
- Distribution des prédictions
- Santé des services

## 🧪 Tests

```bash
# Tests fonctionnels
python test_services.py

# Configuration MLflow
python mlflow_setup.py
```

## 📁 Structure du Projet

```
callcenterai/
├── agent/                 # Service de routage intelligent
├── tfidf_svc/            # Service TF-IDF + SVM
├── transformer_svc/      # Service DistilBERT
├── models/               # Modèles entraînés
├── monitoring/           # Config Prometheus/Grafana
├── docker-compose.yml    # Orchestration services
└── test_services.py     # Suite de tests
```

## 🏷️ Catégories Prédites

| Catégorie | Description | Exemples |
|-----------|-------------|----------|
| **Hardware** | Problèmes matériels | "écran cassé", "laptop broken" |
| **Access** | Accès & authentification | "mot de passe", "login issue" |  
| **Network** | Problèmes réseau | "wifi down", "connexion lente" |
| **Software** | Bugs logiciels | "app crash", "bug application" |

## 🚀 Performance

- **TF-IDF** : ~50ms réponse
- **Transformer** : ~200ms réponse  
- **Agent** : Routage < 10ms
- **Throughput** : 1000+ req/min

## 👥 Équipe

Développé dans le cadre du cours MLOps 2025

## 📄 License

MIT License - Voir LICENSE file
