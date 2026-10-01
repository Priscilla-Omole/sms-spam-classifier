from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="SMS Spam Classifier")

@app.get("/")
def home():
    return {"message": "SMS Spam Classifier API. Visit /docs to try it."}

class Message(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(message: Message):
        spam_probability = model.predict_proba([message.text])[0][1]

        return {
        "label": "spam" if spam_probability >= 0.5 else "ham",
        "spam_probability": round(float(spam_probability), 3),
    }

