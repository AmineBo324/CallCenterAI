import os
import pytest
import joblib

MODEL_PATH = "./models/ticket_classifier_model.pkl"
@pytest.mark.skipif(not os.path.exists('./models/ticket_classifier_model.pkl'), reason="Model file not present in CI")
def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_prediction_output():
    model = joblib.load(MODEL_PATH)
    sample_text = "My laptop won't start"
    prediction = model.predict([sample_text])[0]
    assert prediction in model.classes_
