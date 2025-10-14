import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

# Chemin du dataset
dataset_path = r"C:\Users\OrdiOne\Desktop\MLops\all_tickets_processed_improved_v3.csv"

# Création du dossier fine_tuned_model
output_dir = r"C:\Users\OrdiOne\Desktop\MLops\fine_tuned_model"
os.makedirs(output_dir, exist_ok=True)  

# Chargement du dataset
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset chargé : {len(df)} échantillons.")
except FileNotFoundError:
    print(f"Erreur : Le fichier {dataset_path} n'existe pas.")
    exit()

# Vérification des colonnes
if 'Document' not in df.columns or 'Topic_group' not in df.columns:
    print("Erreur : Le dataset doit contenir les colonnes 'Document' et 'Topic_group'.")
    print(f"Colonnes disponibles : {df.columns.tolist()}")
    exit()

# Nettoyage des données
logging.info("Nettoyage des données...")

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna(subset=['Document', 'Topic_group'])
logging.info(f"Après suppression des valeurs manquantes : {len(df)} échantillons.")

# Supprimer les doublons
df = df.drop_duplicates(subset=['Document'])
logging.info(f"Après suppression des doublons : {len(df)} échantillons.")

# Preparer et Encodage des labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Topic_group'])

# Sauvegarde du LabelEncoder
joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
print(f"LabelEncoder sauvegardé. Classes : {label_encoder.classes_}")

#80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Conversion en Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['Document', 'label']])
test_dataset = Dataset.from_pandas(test_df[['Document', 'label']])
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Nombre de classes
num_labels = len(label_encoder.classes_)
print(f"Nombre de classes : {num_labels}")

# Chargement du tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prétraitement (tokenization)
def preprocess_function(examples):
    return tokenizer(examples['Document'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Chargement du modèle
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Configuration des arguments d'entraînement
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  # Remplacé evaluation_strategy par eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch"
)

# Métrique d'évaluation
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Début du fine-tuning...")
trainer.train()

# Évaluation
results = trainer.evaluate()
print(f"Résultats de l'évaluation : {results}")

# Sauvegarde du modèle et du tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modèle fine-tuné sauvegardé dans : {output_dir}")