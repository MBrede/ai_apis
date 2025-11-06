import os
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Depends
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
from datetime import datetime

from src.core.buffer_class import Model_Buffer
from src.core.config import config


class SentimentBuffer(Model_Buffer):
    """Buffer for sentiment analysis models."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None

    def load_model(self, model_name: str, timeout: int = 300, **kwargs):
        """Load a sentiment analysis model with automatic unloading."""
        # If same model already loaded, just reset timer
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        # Call parent to set up timer
        super().load_model(timeout=timeout)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model_name = model_name
        self.loaded_at = datetime.now()

        # Start timer if configured
        if self.timer:
            self.timer.start()

    def predict_sentiment(self, texts: List[str]) -> List[dict]:
        """Predict sentiment probabilities for input texts."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reset timer on each use
        self.reset_timer()

        if not isinstance(texts, list):
            texts = [texts]

        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = [
            {k: v for k, v in zip(self.model.config.id2label.values(), probability.tolist())}
            for probability in probabilities
        ]
        return probabilities


# Create global buffer instance
sentiment_buffer = SentimentBuffer()

app = FastAPI()
router = APIRouter()

from pydantic import BaseModel
from core.auth import verify_api_key


class Text_Request(BaseModel):
    text: List[str]
    model: str | None = "oliverguhr/german-sentiment-bert"


@router.post("/predict_sentiment/")
async def get_answer(request: Text_Request, api_key: str = Depends(verify_api_key)):
    """Predict sentiment using a sentiment analysis model."""
    sentiment_buffer.load_model(request.model)
    return {"answer": sentiment_buffer.predict_sentiment(request.text)}


@router.get("/buffer_status/")
async def get_buffer_status(api_key: str = Depends(verify_api_key)):
    """Get current buffer status for debugging."""
    return sentiment_buffer.get_status()


app.include_router(router)