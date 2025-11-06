"""
gunicorn llm_wrapper:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 -t 30000
"""

import os
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import requests
import json
from requests.exceptions import ConnectionError, Timeout

class Item(BaseModel):
    instruction: str | None = 'Du bist ein hilfreicher Assistent.'
    input: str | None = 'Erzähl mir eine Geschichte über Schnabeltiere und Data Science.'
    temperature: float | None = Field(0.1, gt=0, leq=1, desc='Temperatur des Modells')
    max_tokens: int | None = Field(500, gt=0, desc='Max Tokens (inklusive Prompt)')
    API: str = 'mistral7b'

class Animal(BaseModel):
    age: int = Field(gt=0, le=10, description="Age of the animal")
    species: str = Field(description="Species of the animal")
    name: str = Field(description="Name of the animal")

class JSON_request(BaseModel):
    prompt: str | None = "Parse the following input as JSON: {input}"
    input: str | None = "Ein Schnabeltier namens Bert, das viel zu viel Zeit mit Rauchen für sein zartes Alter von 4 verbringt"
    schema: str | None = Animal.schema_json()
    API: str = 'mistral7b'


with open('available_endpoints.json', 'r') as f:
    available_endpoints = json.load(f)

class LLM(BaseModel):
    name: str
    source: str
    IP: str
    Port: int
    Unsloth: bool
    Worker_Count: int

def ask_model(item = Item):
    if item.temperature == 0:
        item.temperature = 1e-5
    model = available_endpoints.get('LLM').get(item.API)
    address = f"http://{model.get('IP','149.222.209.66')}:{model.get('Port','8080')}/answer"
    response = requests.post(address,
                             json={'system': item.instruction,
                                   'messages': item.input,
                                   'temperature': item.temperature,
                                   'max_tokens': item.max_tokens})
    if not response.ok:
        return HTTPException(response.status_code, detail=f"Error: {response.text}")
    return response.json()

def ask_for_json(json_request: JSON_request):
    model = available_endpoints.get('LLM').get(json_request.API)
    if model is None or model.get("Unsloth", True):
        return HTTPException(406, detail=f"Error: model is Unsloth or not available")
    address = f"http://{model.get('IP', '149.222.209.66')}:{model.get('Port', '8080')}/json_answer"
    response = requests.post(address,
                             json={'prompt': json_request.prompt,
                                   'input': json_request.input,
                                   'schema': json_request.schema})
    if not response.ok:
        return HTTPException(response.status_code, detail=f"Error: {response.text}")
    return response.json()


app = FastAPI()
router = APIRouter()

@router.get("/list_available_llms")
async def list_available_llms():
    global available_endpoints
    with open('available_endpoints.json', 'r') as f:
        available_endpoints = json.load(f)
    out = {'LLM': {}, 'Image Generation': available_endpoints['Image Generation']}
    for item, content in zip(available_endpoints.get('LLM').keys(),
                             available_endpoints.get('LLM').values()):
        url = (f"http://{content.get('IP')}:"
               f"{content.get('Port')}/ping")
        try:
            if requests.get(url, timeout=1).ok:
                out['LLM'][item] = content
        except (ConnectionError, Timeout):
            pass
    available_endpoints = out
    return out

@router.post("/register_llm")
async def register_llm(llm: LLM):
    with open('available_endpoints.json', 'r') as f:
        available_endpoints = json.load(f)
    available_endpoints['LLM'][llm.name] = llm.dict()
    with open('available_endpoints.json', 'w') as f:
        json.dump(available_endpoints, f)

@router.post("/llm_answer/")
def get_answer(item: Item):
    return ask_model(item)

@router.post("/json_answer/")
def get_json(json_answer: JSON_request):
    return ask_for_json(json_answer)
app.include_router(router)
