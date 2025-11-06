"""
Whisper Audio Transcription API with speaker diarization.

To start:
    gunicorn whisper_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 -t 30000
"""

import os
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import torch
import whisper
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile
from pyannote.audio import Pipeline

from src.core.auth import verify_api_key
from src.core.buffer_class import Model_Buffer
from src.core.config import config


class WhisperBuffer(Model_Buffer):
    """Buffer for Whisper transcription model with automatic unloading."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None

    def load_model(self, model_name: str, timeout: int = 300, **kwargs):
        """Load Whisper model with automatic unloading after timeout."""
        # If same model already loaded, just reset timer
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        # Call parent to set up timer
        super().load_model(timeout=timeout)

        # Load Whisper model
        self.model = whisper.load_model(model_name, **kwargs)
        self.model_name = model_name
        self.loaded_at = datetime.now()

        # Start timer if configured
        if self.timer:
            self.timer.start()

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """Transcribe audio file."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reset timer on each use
        self.reset_timer()

        return self.model.transcribe(audio_path, **kwargs)


class DiarizationBuffer(Model_Buffer):
    """Buffer for pyannote speaker diarization pipeline."""

    def __init__(self):
        super().__init__()

    def load_model(self, timeout: int = 300, **kwargs):
        """Load diarization pipeline with automatic unloading after timeout."""
        # If already loaded, just reset timer
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        # Call parent to set up timer
        super().load_model(timeout=timeout)

        # Load diarization pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=config.HF_TOKEN
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.loaded_at = datetime.now()

        # Start timer if configured
        if self.timer:
            self.timer.start()

    def diarize(
        self,
        audio_path: str,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """Perform speaker diarization on audio file."""
        if not self.is_loaded():
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        # Reset timer on each use
        self.reset_timer()

        if num_speakers is not None:
            return self.pipeline(audio_path, num_speakers=num_speakers)
        elif min_speakers is not None:
            return self.pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        else:
            raise ValueError(
                "Either num_speakers or min_speakers and max_speakers must be provided."
            )


# Create global buffer instances
whisper_buffer = WhisperBuffer()
diarization_buffer = DiarizationBuffer()

# Initialize models
whisper_buffer.load_model(config.DEFAULT_WHISPER_MODEL)
diarization_buffer.load_model()

app = FastAPI()
router = APIRouter()


def diarize_audio(
    file, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None
):
    """Helper function to diarize audio and transcribe each speaker segment."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        mono = tmp.name
        cmd = 'ffmpeg -i "{}" -y -ac 1 {}'.format(file, mono)
        subprocess.check_output(cmd, shell=True)

        # Use diarization buffer
        diarization = diarization_buffer.diarize(
            mono, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers
        )

        # dump the diarization output to disk using RTTM format
        with open("audio.rttm", "w") as rttm:
            diarization.write_rttm(rttm)

    with open("audio.rttm", "r") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        _, _, _, start, duration, _, _, speaker, _, _ = line.split()
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            cmd = f'ffmpeg -ss {start} -i "{file}" -t {duration} -y -ac 1 {tmp.name}'
            subprocess.check_output(cmd, shell=True)

            # Use whisper buffer for transcription
            transcription = whisper_buffer.transcribe(tmp.name, verbose=False)
            out.append(
                {
                    "SPEAKER": speaker,
                    "START": start,
                    "DURATION": duration,
                    "TRANSCRIPTION": transcription["text"],
                    "LANGUAGE": transcription["language"],
                }
            )
    return out


@router.post("/transcribe/")
async def transcribe(
    file: UploadFile, model_to_use: str = "turbo", api_key: str = Depends(verify_api_key)
):
    """Transcribe audio file using Whisper."""
    # Load the requested model (will reuse if already loaded)
    whisper_buffer.load_model(model_to_use)

    # Save uploaded file temporarily
    with open(file.filename, "wb") as f:
        file_contents = await file.read()
        f.write(file_contents)

    # Transcribe using buffer
    answer = whisper_buffer.transcribe(file.filename, verbose=False)["text"]
    os.remove(file.filename)
    return {"answer": answer}


@router.post("/transcribe_and_diarize/")
async def transcribe_diarize(
    file: UploadFile,
    model_to_use: str = "turbo",
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    api_key: str = Depends(verify_api_key),
):
    """Transcribe audio with speaker identification."""
    # Load the requested Whisper model
    whisper_buffer.load_model(model_to_use)

    # Ensure diarization pipeline is loaded
    diarization_buffer.load_model()

    # Save uploaded file temporarily
    with open(file.filename, "wb") as f:
        file_contents = await file.read()
        f.write(file_contents)

    try:
        answer = diarize_audio(
            file.filename,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        os.remove(file.filename)
        return {"answer": answer}
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="You need to specify either num_speakers or both min_speakers and max_speakers!",
        )


@router.get("/buffer_status/")
async def get_buffer_status(api_key: str = Depends(verify_api_key)):
    """Get current buffer status for debugging."""
    return {"whisper": whisper_buffer.get_status(), "diarization": diarization_buffer.get_status()}


app.include_router(router)
