"""
Whisper Audio Transcription API with speaker diarization.

To start:
    gunicorn whisper_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 -t 30000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import subprocess
import numpy as np
import tempfile
import torch

import whisper
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Depends
from pyannote.audio import Pipeline

from config import config
from auth import verify_api_key

loaded_model = config.DEFAULT_WHISPER_MODEL
model = whisper.load_model(loaded_model)
app = FastAPI()
router = APIRouter()

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=config.HF_TOKEN).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def diarize(file, 
            num_speakers: int = None,
            min_speakers: int = None,
            max_speakers: int = None):
    global model
    
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        mono = tmp.name
        cmd = 'ffmpeg -i "{}" -y -ac 1 {}'.format(file, mono)
        subprocess.check_output(cmd, shell=True)
        if num_speakers is not None:
            diarization = pipeline(mono, num_speakers=num_speakers)
        elif min_speakers is not None:
            diarization = pipeline(mono, min_speakers=min_speakers, 
                                    max_speakers=max_speakers)
        else:
            raise ValueError("Either num_speakers or min_speakers and max_speakers must be provided.")

        # dump the diarization output to disk using RTTM format
        with open("audio.rttm", "w") as rttm:
            diarization.write_rttm(rttm)
    with open ("audio.rttm", "r") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        _, _, _, start, duration, _, _, speaker, _, _ = line.split()
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            cmd = f'ffmpeg -ss {start} -i "{file}" -t {duration} -y -ac 1 {tmp.name}'
            subprocess.check_output(cmd, shell=True)
            transcription = model.transcribe(tmp.name, verbose=False)
            out.append(
                {
                    'SPEAKER': speaker,
                    'START': start,
                    'DURATION': duration,
                    'TRANSCRIPTION': transcription['text'],
                    'LANGUAGE': transcription['language']
                }
            )
    return out


@router.post("/transcribe/")
async def transcribe(file: UploadFile, model_to_use: str = 'turbo', api_key: str = Depends(verify_api_key)):
    """Transcribe audio file using Whisper."""
    global model
    global loaded_model
    if loaded_model != model_to_use:
        loaded_model = model_to_use
        model = whisper.load_model(loaded_model)
    with open(file.filename, 'wb') as f:
        file_contents = await file.read()
        f.write(file_contents)
    answer = model.transcribe(file.filename, verbose=False)['text']
    os.remove(file.filename)
    return {"answer": answer}

@router.post("/transcribe_and_diarize/")
async def transcribe_diarize(file: UploadFile, model_to_use: str = 'turbo',
                     num_speakers: int = None,
                     min_speakers : int = None,
                     max_speakers : int = None,
                     api_key: str = Depends(verify_api_key)):
    """Transcribe audio with speaker identification."""
    global model
    global loaded_model
    if loaded_model != model_to_use:
        loaded_model = model_to_use
        model = whisper.load_model(loaded_model)
    with open(file.filename, 'wb') as f:
        file_contents = await file.read()
        f.write(file_contents)
    try:
        answer = diarize(file.filename, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        os.remove(file.filename)
        return {"answer": answer}
    except ValueError:
        raise HTTPException(status_code=404, detail="You need to specify either num_speakers or both min_speakers and max_speakers!")


app.include_router(router)