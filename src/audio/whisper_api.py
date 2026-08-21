"""
Whisper Audio Transcription API with speaker diarization.

Model/buffer classes live in `src.audio.stt_backends` (one module per
backend, shared base classes for the two common shapes — see
`stt_backends/base.py`). This file is just the FastAPI routing layer:
endpoints, request/response shaping, and legacy per-segment orchestration.

To start:
    gunicorn whisper_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 -t 30000
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from collections import Counter

from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.audio.stt_backends import (
    BACKEND_REGISTRY,
    FILLER_PROMPT,
    diarization_buffer,
    whisper_buffer,
)
from src.core.app_factory import create_app
from src.core.auth_dependencies import verify_api_key

logger = logging.getLogger(__name__)


def _words_per_second(text: str, duration: float) -> float:
    if duration <= 0:
        return float("inf")
    return len(text.split()) / duration


def filter_transcription_chunks(
    chunks: list[dict],
    max_words_per_second: float | None = 6.0,
    top_n_languages: int | None = 2,
) -> list[dict]:
    """Filter transcription chunks by word rate and language.

    Args:
        chunks: List of transcription dicts with TRANSCRIPTION, DURATION, LANGUAGE keys.
        max_words_per_second: Hard ceiling on word rate. Human speech peaks at ~6 WPS
            (360 WPM); anything above is Whisper hallucination. Set None to disable.
        top_n_languages: Keep only chunks whose detected language is among the N most
            common languages in the batch. Set None to disable.
    """
    if not chunks:
        return chunks

    result = list(chunks)

    if max_words_per_second is not None:
        before = len(result)
        result = [
            c for c in result
            if _words_per_second(c["TRANSCRIPTION"], c["DURATION"]) <= max_words_per_second
        ]
        logger.info(
            "Word-rate filter (max %.1f WPS): removed %d/%d chunks",
            max_words_per_second, before - len(result), before,
        )

    if top_n_languages is not None and result:
        lang_counts = Counter(c["LANGUAGE"] for c in result)
        allowed = {lang for lang, _ in lang_counts.most_common(top_n_languages)}
        before = len(result)
        result = [c for c in result if c["LANGUAGE"] in allowed]
        logger.info(
            "Language filter (top %d: %s): removed %d/%d chunks",
            top_n_languages, allowed, before - len(result), before,
        )

    return result


app = create_app(
    title="Whisper Transcription API",
    description="Speech-to-text with speaker diarization.",
)
router = APIRouter()


def diarize_audio(
    file, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None
):
    """Diarize audio and transcribe each speaker segment (legacy per-segment approach)."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        mono = tmp.name
        cmd = f'ffmpeg -i "{file}" -y -ac 1 {mono}'
        subprocess.check_output(cmd, shell=True)

        diarization = diarization_buffer.diarize(
            mono, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers
        )

        lines = diarization.serialize()["diarization"]

    out = []
    for line in lines:
        start, end, speaker = line.values()
        duration = end - start
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            cmd = f'ffmpeg -ss {start} -i "{file}" -t {duration} -y -ac 1 {tmp.name}'
            subprocess.check_output(cmd, shell=True)

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
    if not whisper_buffer.is_loaded() or whisper_buffer.model_name != model_to_use:
        logger.info(f"Loading Whisper model on request: {model_to_use}")
        await asyncio.to_thread(whisper_buffer.load_model, model_to_use)

    with open(file.filename, "wb") as f:
        file_contents = await file.read()
        f.write(file_contents)

    result = await asyncio.to_thread(whisper_buffer.transcribe, file.filename, verbose=False)
    answer = result["text"]
    os.remove(file.filename)
    return {"answer": answer}


@router.post("/transcribe_and_diarize/")
async def transcribe_diarize(
    file: UploadFile,
    model_to_use: str = "turbo",
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    max_words_per_second: float | None = 6.0,
    top_n_languages: int | None = 2,
    backend: str = "whisperx",
    align: bool = True,
    include_fillers: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """Transcribe audio with speaker identification.

    Args:
        backend: "whisperx" (default) transcribes the full audio in one batched pass,
            then aligns word timestamps and assigns speakers — far less hallucination.
            "qwen3-asr" uses Qwen3-ASR-1.7B + Qwen3-ForcedAligner instead of
            Whisper/WhisperX. "granite-speech" (ibm-granite/granite-speech-4.1-2b-plus,
            supports German) uses its own word-timestamp prompt, same
            align-then-diarize shape as qwen3-asr. "ark-asr" (Audio8/ARK-ASR-3B),
            "hojo-asr" (HojoAI/Hojo-ASR-V1, no German support), and "nemotron-asr"
            (nvidia/nemotron-3.5-asr-streaming-0.6b) diarize first and transcribe
            each speaker turn separately, since none of those three expose
            word-level timestamps. See `src.audio.stt_backends` for each
            backend's implementation. "whisper" uses the legacy per-segment
            approach (pyannote → ffmpeg → whisper).
        max_words_per_second: Remove chunks whose word rate exceeds this value.
            Human speech peaks at ~6 WPS; higher values are Whisper hallucinations.
            Set to 0 to disable.
        top_n_languages: Keep only the N most common detected languages across all
            chunks. Set to 0 to disable.
        num_speakers: (legacy backend only) exact speaker count.
            WhisperX backend accepts this as a hint but can auto-detect.
        include_fillers: Inject a prompt nudging Whisper to retain filler words
            (ähm, uhm, erm, etc.) that it would otherwise suppress.
            Only effective with the whisperx backend — ignored (with a
            warning) for every other backend, none of which have an
            equivalent prompt-injection mechanism.
    """
    wps_limit = max_words_per_second if max_words_per_second and max_words_per_second > 0 else None
    lang_limit = top_n_languages if top_n_languages and top_n_languages > 0 else None
    filler_prompt = FILLER_PROMPT if include_fillers else None

    with open(file.filename, "wb") as f:
        file_contents = await file.read()
        f.write(file_contents)

    try:
        if backend in BACKEND_REGISTRY:
            buf = BACKEND_REGISTRY[backend]
            if include_fillers and backend != "whisperx":
                logger.warning("include_fillers has no effect with backend=%s — ignoring.", backend)

            await asyncio.to_thread(buf.ensure_loaded, model_to_use)

            chunks = await asyncio.to_thread(
                buf.transcribe_and_diarize,
                file.filename,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                align=align,
                initial_prompt=filler_prompt if backend == "whisperx" else None,
            )
        else:
            if not whisper_buffer.is_loaded() or whisper_buffer.model_name != model_to_use:
                logger.info(f"Loading Whisper model on request: {model_to_use}")
                await asyncio.to_thread(whisper_buffer.load_model, model_to_use)

            if not diarization_buffer.is_loaded():
                logger.info("Loading diarization pipeline on request")
                await asyncio.to_thread(diarization_buffer.load_model)

            chunks = await asyncio.to_thread(
                diarize_audio,
                file.filename,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        os.remove(file.filename)
        answer = filter_transcription_chunks(chunks, max_words_per_second=wps_limit, top_n_languages=lang_limit)
        return {"answer": answer, "backend": backend, "removed_chunks": len(chunks) - len(answer)}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/buffer_status/")
async def get_buffer_status(api_key: str = Depends(verify_api_key)):
    """Get current buffer status for debugging."""
    status = {
        "whisper": whisper_buffer.get_status(),
        "diarization": diarization_buffer.get_status(),
    }
    status.update({name.replace("-", "_"): buf.get_status() for name, buf in BACKEND_REGISTRY.items()})
    return status


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Docker HEALTHCHECK.
    Tests if API is running and buffers are functioning.
    Returns 200 OK when healthy (ready to accept requests).
    Note: Models load on first request (lazy loading).
    """
    logger.info("=== WHISPER HEALTH CHECK STARTED ===")
    try:
        statuses = {
            "whisper": whisper_buffer.get_status(),
            "diarization": diarization_buffer.get_status(),
        }
        statuses.update({name.replace("-", "_"): buf.get_status() for name, buf in BACKEND_REGISTRY.items()})

        healthy = {name: status is not None for name, status in statuses.items()}
        is_healthy = all(healthy.values())

        response_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "whisper-api",
            **{f"{name}_buffer_accessible": ok for name, ok in healthy.items()},
            **{
                f"{name}_model_loaded": status.get("is_loaded", False) if status else False
                for name, status in statuses.items()
            },
            "note": "Models will load on first request",
        }

        if not is_healthy:
            return JSONResponse(status_code=503, content=response_data)

        logger.info("=== WHISPER HEALTH CHECK COMPLETED SUCCESSFULLY ===")
        return response_data

    except Exception as e:
        logger.error(f"Whisper health check failed with exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "whisper-api", "error": str(e)},
        )


app.include_router(router)
