from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List

tokenizer = None
model = None

def set_model_and_tokenizer(model_name):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(texts):
    if not isinstance(texts, list):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = [{k: v for k,v in zip(model.config.id2label.values(), 
                                            probability.tolist())}
                     for probability in probabilities]
    return probabilities

app = FastAPI()
router = APIRouter()

from pydantic import BaseModel


class Text_Request(BaseModel):
    text: List[str]
    model: str | None = "oliverguhr/german-sentiment-bert"


@router.post("/predict_sentiment/")
async def get_answer(request: Text_Request):
    set_model_and_tokenizer(request.model)
    return {"answer": predict_sentiment(request.text)}


app.include_router(router)