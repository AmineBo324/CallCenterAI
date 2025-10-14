# ============================================
# BLOC 1: IMPORTS ET CONFIGURATION
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
# ============================================
# BLOC 2: CHARGEMENT DES DONNÉES
# ============================================
df = pd.read_csv('/kaggle/input/tickets/all_tickets_processed_improved_v3.csv')
print("✅ Données chargées!")
print(f"Shape: {df.shape}")
print(df.head())
# ============================================
# BLOC 3: EXPLORATION DES DONNÉES
# ============================================
print("\n📊 Informations sur le dataset:")
print(f"Nombre total de tickets: {len(df)}")
print(f"Colonnes: {list(df.columns)}")

print("\n🔍 Valeurs manquantes:")
print(df.isnull().sum())

print("\n📈 Distribution des catégories:")
print(df['Topic_group'].value_counts())
print(f"\nPourcentages:")
print(df['Topic_group'].value_counts(normalize=True) * 100)
# ============================================
# BLOC 4: VISUALISATION DE LA DISTRIBUTION
# ============================================
plt.subplot(1, 2, 1)
df['Topic_group'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribution des Catégories', fontsize=14, fontweight='bold')
plt.xlabel('Catégorie', fontsize=11)
plt.ylabel('Nombre de tickets', fontsize=11)
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
df['Topic_group'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Répartition en %', fontsize=14, fontweight='bold')
plt.ylabel('')

plt.tight_layout()
plt.show()

print("✅ Visualisation créée!")

# ============================================
# BLOC 5: PRÉPARATION DES DONNÉES
# ============================================
X = df['Document']
y = df['Topic_group']

print(f"✅ Features (X): {len(X)} documents")
print(f"✅ Target (y): {len(y)} catégories")
print(f"\nCatégories uniques: {y.unique()}")
# ============================================
# BLOC 6: DIVISION TRAIN/TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% pour le test
    random_state=42,     # Pour la reproductibilité
    stratify=y           # Garde les mêmes proportions
)

print("✅ Données divisées:")
print(f"\n📚 Ensemble d'entraînement: {len(X_train)} tickets ({len(X_train)/len(df)*100:.1f}%)")
print(f"🧪 Ensemble de test: {len(X_test)} tickets ({len(X_test)/len(df)*100:.1f}%)")

print("\n📊 Distribution dans l'entraînement:")
print(y_train.value_counts())

print("\n📊 Distribution dans le test:")
print(y_test.value_counts())
# ============================================
# BLOC 7: PIPELINE AVEC PROBABILITÉS
# ============================================

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.7,
        sublinear_tf=True
    )),
    ('svm', CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=42, max_iter=1000, dual=False),
        cv=3  # Ajoute les probabilités
    ))
])

print("✅ Pipeline créé avec probabilités!")
print(model)
# ============================================
# BLOC 8: ENTRAÎNEMENT DU MODÈLE
# ============================================

print("🔄 Démarrage de l'entraînement...")
print("⏳ Veuillez patienter...")

# Entraînement
model.fit(X_train, y_train)

print("\n✅ Entraînement terminé avec succès!")
print(f"✅ Modèle entraîné sur {len(X_train)} tickets")
# ============================================
# BLOC 9: PRÉDICTIONS
# ============================================

print("🔮 Génération des prédictions...")

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"✅ {len(y_pred)} prédictions générées!")

# Afficher quelques exemples
print("\n📝 Exemples de prédictions:")
for i in range(min(5, len(X_test))):
    print(f"\n{i+1}. Document: {X_test.iloc[i][:60]}...")
    print(f"   Vraie catégorie: {y_test.iloc[i]}")
    print(f"   Prédiction: {y_pred[i]}")
    print(f"   Confiance: {max(y_pred_proba[i]):.2%}")
# ============================================
# BLOC 10: ÉVALUATION - ACCURACY
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("="*60)
print("📊 RÉSULTATS DE L'ÉVALUATION")
print("="*60)
print(f"\n🎯 Accuracy globale: {accuracy:.2%}")
print(f"   → {int(accuracy * len(y_test))} prédictions correctes sur {len(y_test)}")

# Calculer l'accuracy par catégorie
print("\n📈 Accuracy par catégorie:")
for category in sorted(y_test.unique()):
    mask = y_test == category
    if mask.sum() > 0:
        cat_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        print(f"   {category:15s}: {cat_accuracy:.2%}")
# ============================================
# BLOC 11: FONCTION DE PRÉDICTION
# ============================================

def predict_ticket(text, show_probas=True):
    
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    
    print(f"\n{'='*60}")
    print(f"📝 Ticket: {text[:70]}...")
    print(f"✅ Catégorie prédite: {prediction}")
    print(f"🎯 Confiance: {max(probabilities):.2%}")
    
    if show_probas:
        print(f"\n📊 Probabilités détaillées:")
        sorted_idx = np.argsort(probabilities)[::-1]
        for idx in sorted_idx:
            cls = model.classes_[idx]
            prob = probabilities[idx]
            bar = "█" * int(prob * 40)
            print(f"   {cls:15s}: {prob:6.2%} {bar}")
    
    return prediction

print("✅ Fonction predict_ticket() créée!")
print("\n💡 Utilisation: predict_ticket('votre texte ici')")

# ============================================
# BLOC 12: SAUVEGARDE DU MODÈLE
# ============================================
import os
# Sauvegarder le modèle
filename = 'ticket_classifier_model.pkl'
joblib.dump(model, filename)

print(f"✅ Modèle sauvegardé dans '{filename}'")
print(f"📦 Taille du fichier: {os.path.getsize(filename) / 1024:.2f} KB")

print("\n💡 Pour charger le modèle plus tard:")
print("   model_loaded = joblib.load('ticket_classifier_model.pkl')")
