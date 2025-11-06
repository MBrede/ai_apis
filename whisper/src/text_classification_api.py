from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_api
import torch
from typing import List
from dotenv import load_dotenv
import os
from setfit import SetFitModel

load_dotenv(".env")
tokenizer = None
model = None

def set_model_and_tokenizer(model_name):
    global tokenizer, model
    model_info = hf_api.model_info(model_name, token  = os.getenv("hf_token"))
    if model_info.library_name == "setfit":
        model = SetFitModel.from_pretrained(model_name)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token = os.getenv("hf_token"))
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token = os.getenv("hf_token"))

def predict_proba(texts):
    if not isinstance(texts, list):
        texts = [texts]
    if tokenizer is None:
        probabilities = model.predict_proba(texts)
        probabilities = [{k: v for k,v in zip(model.id2label.values(), 
                                                probability.tolist())}
                        for probability in probabilities]
    else:
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


@router.post("/predict_proba/")
async def get_answer(request: Text_Request):
    set_model_and_tokenizer(request.model)
    return {"answer": predict_proba(request.text)}


app.include_router(router)
