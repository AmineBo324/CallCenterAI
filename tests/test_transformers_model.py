import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

MODEL_DIR = "./models/fine_tuned_model"

def test_transformer_model_loads():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    assert tokenizer is not None
    assert model is not None
    assert label_encoder is not None

def test_transformer_prediction():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    model.eval()
    inputs = tokenizer("My computer is broken", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(probs, dim=-1).item()
    label = label_encoder.inverse_transform([predicted_class_id])[0]
    assert label in label_encoder.classes_
