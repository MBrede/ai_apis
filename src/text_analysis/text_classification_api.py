from datetime import datetime

import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from huggingface_hub import hf_api
from setfit import SetFitModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.buffer_class import Model_Buffer
from src.core.config import config

load_dotenv(".env")


class ClassificationBuffer(Model_Buffer):
    """Buffer for text classification models with support for both SetFit and standard models."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None
        self.is_setfit: bool = False

    def load_model(self, model_name: str, timeout: int = 300, **kwargs):
        """Load a text classification model with automatic unloading."""
        # If same model already loaded, just reset timer
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        # Call parent to set up timer
        super().load_model(timeout=timeout)

        # Determine model type
        model_info = hf_api.model_info(model_name, token=config.HF_TOKEN)
        self.is_setfit = model_info.library_name == "setfit"

        # Load model based on type
        if self.is_setfit:
            self.model = SetFitModel.from_pretrained(model_name)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.HF_TOKEN)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, token=config.HF_TOKEN
            )

        self.model_name = model_name
        self.loaded_at = datetime.now()

        # Start timer if configured
        if self.timer:
            self.timer.start()

    def predict_proba(self, texts: list[str]) -> list[dict]:
        """Predict class probabilities for input texts."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reset timer on each use
        self.reset_timer()

        if not isinstance(texts, list):
            texts = [texts]

        if self.is_setfit:
            # SetFit model prediction
            probabilities = self.model.predict_proba(texts)
            probabilities = [
                {k: v for k, v in zip(self.model.id2label.values(), probability.tolist())}
                for probability in probabilities
            ]
        else:
            # Standard transformer model prediction
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
classification_buffer = ClassificationBuffer()

app = FastAPI()
router = APIRouter()

from fastapi import Depends
from pydantic import BaseModel

from src.core.auth import verify_api_key


class Text_Request(BaseModel):
    text: list[str]
    model: str | None = "oliverguhr/german-sentiment-bert"


@router.post("/predict_proba/")
async def get_answer(request: Text_Request, api_key: str = Depends(verify_api_key)):
    """Predict class probabilities using a text classification model."""
    classification_buffer.load_model(request.model)
    return {"answer": classification_buffer.predict_proba(request.text)}


@router.get("/buffer_status/")
async def get_buffer_status(api_key: str = Depends(verify_api_key)):
    """Get current buffer status for debugging."""
    return classification_buffer.get_status()


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Docker HEALTHCHECK.
    Tests if API is running and buffer is functioning.
    Returns 200 OK when healthy, 503 Service Unavailable when unhealthy.
    """
    try:
        # Test if buffer is accessible and working
        buffer_status = classification_buffer.get_status()

        # Check if we can access buffer attributes
        is_healthy = buffer_status is not None

        response_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "text-classification-api",
            "buffer_accessible": is_healthy,
            "model_loaded": buffer_status.get("is_loaded", False) if buffer_status else False,
        }

        # Return 503 if unhealthy, 200 if healthy
        if not is_healthy:
            return JSONResponse(status_code=503, content=response_data)
        return response_data

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "text-classification-api",
                "error": str(e),
            },
        )


app.include_router(router)
