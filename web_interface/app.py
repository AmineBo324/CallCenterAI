from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import time
from datetime import datetime

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration simple des services (URLs fixes)
SERVICES = {
    'tfidf': 'http://localhost:8001',
    'transformer': 'http://localhost:8002',
    'agent': 'http://localhost:8000'
}

# Feature flags simples
FEATURE_FLAGS = {
    'claude_haiku_4_5': False
}

# Statistiques globales
stats = {
    'total_predictions': 0,
    'total_latency': 0,
    'categories_count': {},
    'service_usage': {},
    'start_time': datetime.now()
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        service = data.get('service', 'agent')

        if not text:
            return jsonify({'error': 'Texte requis'}), 400

        start_time = time.time()

        # Vérifier le service demandé
        if service not in SERVICES:
            return jsonify({'error': 'Service non supporté'}), 400
        url = f"{SERVICES[service]}/predict"

        payload = {'text': text}
        if data.get('use_model') == 'claude_haiku_4_5':
            if not FEATURE_FLAGS['claude_haiku_4_5']:
                return jsonify({'error': 'Model claude_haiku_4_5 not enabled'}), 403
            payload['force_model'] = 'claude_haiku_4_5'

        response = requests.post(url, json=payload, timeout=30)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()

            # Standardiser les clés pour le frontend
            if isinstance(result, dict):
                if 'label' in result:
                    result['category'] = result.pop('label')
                if 'confidence' not in result:
                    result['confidence'] = 1.0
                if 'model_used' not in result:
                    result['model_used'] = service
                if 'detected_language' not in result:
                    result['detected_language'] = 'unknown'
                if 'pii_detected' not in result:
                    result['pii_detected'] = False
                if 'explanation' not in result:
                    result['explanation'] = ''

            update_stats(result if isinstance(result, dict) else {}, service, latency)

            out = dict(result)
            out['latency_ms'] = round(latency, 2)
            out['timestamp'] = datetime.now().isoformat()
            return jsonify(out)

        else:
            return jsonify({'error': f'Erreur du service {service}: {response.status_code}'}), response.status_code

    except requests.RequestException as e:
        return jsonify({'error': f'Erreur de connexion au service {service}: {str(e)}'}), 503
    except Exception as e:
        return jsonify({'error': f'Erreur interne: {str(e)}'}), 500

@app.route('/api/health')
def health():
    status = {}
    for name, url in SERVICES.items():
        try:
            r = requests.get(f"{url}/health", timeout=5)
            status[name] = 'healthy' if r.status_code == 200 else 'unhealthy'
        except Exception as e:
            status[name] = f'unreachable ({e})'
    return jsonify({'web_interface': 'healthy', 'services': status, 'timestamp': datetime.now().isoformat()})

@app.route('/api/flags', methods=['GET', 'POST'])
def flags():
    if request.method == 'GET':
        return jsonify(FEATURE_FLAGS)
    data = request.get_json() or {}
    for k, v in data.items():
        if k in FEATURE_FLAGS:
            FEATURE_FLAGS[k] = bool(v)
    return jsonify(FEATURE_FLAGS)

@app.route('/api/stats')
def get_stats():
    uptime = datetime.now() - stats['start_time']
    avg_latency = (stats['total_latency'] / stats['total_predictions']) if stats['total_predictions'] > 0 else 0
    most_common = max(stats['categories_count'].items(), key=lambda x: x[1])[0] if stats['categories_count'] else None
    return jsonify({
        'total_predictions': stats['total_predictions'],
        'avg_latency_ms': round(avg_latency,2),
        'categories_count': stats['categories_count'],
        'service_usage': stats['service_usage'],
        'most_common_category': most_common,
        'uptime_seconds': uptime.total_seconds(),
        'uptime_readable': str(uptime).split('.')[0]
    })

def update_stats(prediction, service, latency):
    stats['total_predictions'] += 1
    stats['total_latency'] += latency

    # Catégorie
    category = 'unknown'
    if isinstance(prediction, dict):
        category = prediction.get('category') or prediction.get('label') or 'unknown'
    elif isinstance(prediction, str):
        category = prediction
    stats['categories_count'][category] = stats['categories_count'].get(category, 0) + 1

    # Service usage
    stats['service_usage'][service] = stats['service_usage'].get(service, 0) + 1

if __name__ == '__main__':
    print("🌐 CallCenterAI Web Interface démarrée sur http://localhost:5001")
    print("🔄 Services attendus:")
    for s, url in SERVICES.items():
        print(f"   • {s}: {url}")
    app.run(debug=True, host='0.0.0.0', port=5001)
