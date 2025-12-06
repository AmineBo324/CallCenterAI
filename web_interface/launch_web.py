#!/usr/bin/env python3
"""
Script de lancement de l'interface web CallCenterAI
Usage: python launch_web.py
"""

import subprocess
import sys
import os
import time
import requests
import webbrowser
from datetime import datetime

def print_banner():
    """Afficher le banner de l'application"""
    print("=" * 60)
    print("🤖 CALLCENTERAI - INTERFACE WEB")
    print("=" * 60)
    print(f"⏰ Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_python_environment():
    """Vérifier l'environnement Python"""
    print("🔍 Vérification de l'environnement Python...")
    
    # Vérifier la version de Python
    print(f"   ✅ Python {sys.version.split()[0]}")
    
    # Vérifier les packages requis
    required_packages = ['flask', 'flask_cors', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (manquant)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Packages manquants: {', '.join(missing_packages)}")
        print("📦 Installation automatique...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"   ✅ {package} installé")
            except subprocess.CalledProcessError:
                print(f"   ❌ Échec d'installation de {package}")
                return False
    
    print("✅ Environnement Python configuré\n")
    return True

def check_services():
    """Vérifier l'état des services microservices"""
    print("🔍 Vérification des services...")
    
    services = {
        'TF-IDF Service': 'http://localhost:8001/health',
        'Transformer Service': 'http://localhost:8002/health', 
        'Agent Service': 'http://localhost:8000/health'
    }
    
    available_services = []
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"   ✅ {service_name}")
                available_services.append(service_name)
            else:
                print(f"   ⚠️  {service_name} (réponse: {response.status_code})")
        except requests.RequestException:
            print(f"   ❌ {service_name} (non accessible)")
    
    if not available_services:
        print("\n⚠️  Aucun service microservice détecté")
        print("💡 L'interface web fonctionnera en mode démo")
        print("📌 Pour activer tous les services, lancez:")
        print("   python start.py")
    else:
        print(f"\n✅ {len(available_services)} service(s) disponible(s)")
    
    return len(available_services)

def start_web_interface():
    """Démarrer l'interface web Flask"""
    print("🚀 Lancement de l'interface web...")
    
    # Changer vers le répertoire web_interface
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    
    # Définir les variables d'environnement Flask
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    print("📍 Interface disponible sur: http://localhost:5001")
    print("🔄 Démarrage du serveur Flask...")
    
    # Attendre un peu puis ouvrir le navigateur
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5001')
            print("🌐 Navigateur ouvert automatiquement")
        except Exception as e:
            print(f"⚠️  Impossible d'ouvrir le navigateur: {e}")
    
    # Lancer l'ouverture du navigateur en arrière-plan
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Démarrer Flask
    try:
        from app import app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5001,
            use_reloader=False  # Éviter le double démarrage en mode debug
        )
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt de l'interface web")
    except Exception as e:
        print(f"\n❌ Erreur lors du démarrage: {e}")
        return False
    
    return True

def show_usage_instructions():
    """Afficher les instructions d'utilisation"""
    print("\n" + "=" * 60)
    print("📖 INSTRUCTIONS D'UTILISATION")
    print("=" * 60)
    print("1. 📝 Saisissez votre message dans la zone de texte")
    print("2. 🎯 Choisissez un type de prédiction:")
    print("   • ⚡ TF-IDF (Rapide) - Modèle classique")
    print("   • 🧠 Transformer (Précis) - Modèle avancé") 
    print("   • 🎯 Agent (Auto) - Routage intelligent")
    print("3. 📊 Consultez les résultats et statistiques")
    print()
    print("💡 Exemples de messages à tester:")
    print("   • Ma facture est incorrecte")
    print("   • Je n'arrive pas à me connecter")
    print("   • J'ai besoin d'aide pour configurer")
    print("   • Je veux changer d'offre")
    print()
    print("🔧 Raccourcis clavier:")
    print("   • Ctrl + Enter: Prédiction avec l'Agent")
    print("   • F5: Actualiser la page")
    print()

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérifier l'environnement
    if not check_python_environment():
        print("❌ Problème d'environnement Python. Arrêt.")
        sys.exit(1)
    
    # Vérifier les services
    services_count = check_services()
    
    # Afficher les instructions
    show_usage_instructions()
    
    # Démarrer l'interface web
    print("🎬 Appuyez sur Ctrl+C pour arrêter le serveur")
    print("-" * 60)
    
    success = start_web_interface()
    
    if success:
        print("\n✅ Interface web fermée proprement")
    else:
        print("\n❌ Problème lors du démarrage")
        sys.exit(1)

if __name__ == "__main__":
    main()