import requests

tickets = [
    "laptop broken",

    "Bonjour, j’ai un problème avec mon ordinateur qui s’éteint sans raison.",

    """Hello, I'm unable to access the shared financial dashboard even though I have 
    the correct credentials. This issue started after the last system update and 
    affects several members of my department.""",

    "Hi, my email is john@example.com and I forgot my VPN password. Can someone reset it?"
]

for ticket in tickets:
    print(f"\n{'='*80}")
    print(f"📝 Ticket: {ticket}")
    print('='*80)
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": ticket}
    )
    
    result = response.json()
    print(f"✅ Catégorie    : {result['label']}")
    print(f"🎯 Confiance    : {result['confidence']:.2%}")
    print(f"🤖 Modèle       : {result['model_used']}")
    print(f"🌍 Langue       : {result['detected_language']}")
    print(f"🔒 PII détecté  : {result['pii_detected']}")
    print(f"💡 Explication  : {result['explanation']}")
