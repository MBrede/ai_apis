"""
Whisper Audio Transcription API with speaker diarization.

Model/buffer classes live in `src.audio.stt_backends` (one module per
backend, shared base classes for the two common shapes — see
`stt_backends/base.py`). This file is just the FastAPI routing layer:
endpoints, request/response shaping, and legacy per-segment orchestration.

This SAME file runs in three separate containers (ai-apis-whisper,
ai-apis-whisper-qwen3asr, ai-apis-whisper-hojoasr) — see pyproject.toml's
whisper-only/whisper-qwen3asr-only/whisper-hojoasr-only extras. Some
backends have genuinely conflicting dependency pins (qwen-asr needs
transformers==4.57.6, hojo-asr needs torch<2.6, Granite-Speech/ARK-ASR need
transformers>=5.8) that can't coexist in one environment, so each container
only installs (and only runs in-process) a subset of backends, controlled
by the WHISPER_LOCAL_BACKENDS env var. A request for a backend this
container doesn't run locally is proxied to whichever other container does,
via WHISPER_PROXY_URL_<BACKEND> env vars — see `_proxy_transcribe_and_diarize`.

To start:
    gunicorn whisper_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 -t 30000
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter

import aiohttp
from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.audio.stt_backends import (
    BACKEND_REGISTRY,
    FILLER_PROMPT,
    diarization_buffer,
    whisper_buffer,
)
from src.core.app_factory import create_app
from src.core.auth import build_auth_headers
from src.core.auth_dependencies import verify_api_key
from src.core.config import config

logger = logging.getLogger(__name__)

# Which backends this container runs in-process. Defaults to "everything"
# (today's single-container behavior) so nothing breaks for a deployment
# that doesn't set this — each split container overrides it explicitly.
# "whisper" is the legacy pyannote+ffmpeg+plain-whisper fallback, not in
# BACKEND_REGISTRY (see diarize_audio below), so it's included by default here too.
LOCAL_BACKENDS: set[str] = {
    b.strip()
    for b in os.environ.get(
        "WHISPER_LOCAL_BACKENDS", ",".join(BACKEND_REGISTRY) + ",whisper"
    ).split(",")
    if b.strip()
}


def _proxy_url_env_var(backend: str) -> str:
    return f"WHISPER_PROXY_URL_{backend.upper().replace('-', '_')}"


# Backends not run locally, with a configured URL to proxy them to. A
# backend that's neither local nor proxied is simply unavailable (404) —
# e.g. a dev container that only wants a subset without wiring up the rest.
PROXY_URLS: dict[str, str] = {
    name: url
    for name in BACKEND_REGISTRY
    if name not in LOCAL_BACKENDS and (url := os.environ.get(_proxy_url_env_var(name)))
}


async def _proxy_transcribe_and_diarize(target_url: str, file_path: str, params: dict) -> dict:
    """Forward a /transcribe_and_diarize/ request to another whisper-* container and relay its JSON response verbatim.

    The remote container applies its own filter_transcription_chunks (same
    params forwarded through), so the response is returned as-is — no
    local re-filtering.
    """
    timeout = aiohttp.ClientTimeout(total=int(os.environ.get("WHISPER_PROXY_TIMEOUT", "1800")))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with open(file_path, "rb") as fh:
            form = aiohttp.FormData()
            form.add_field("file", fh, filename=os.path.basename(file_path))
            async with session.post(
                f"{target_url.rstrip('/')}/transcribe_and_diarize/",
                params=params,
                data=form,
                headers=build_auth_headers(config.API_KEY),
            ) as resp:
                body = await resp.read()
                if not (200 <= resp.status < 400):
                    raise HTTPException(status_code=resp.status, detail=body.decode(errors="replace"))
                return json.loads(body)


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


def _audio_duration_seconds(audio_path: str) -> float:
    """Return an audio/video file's duration in seconds.

    Used so the caller (sync.py) can flag a transcript whose last segment
    ends well before the real end of the file — a cheap sanity check for
    silent truncation (e.g. a generation-length cap cutting a backend off
    mid-chunk, see qwen3_asr.py's max_new_tokens fix). Reuses
    `whisperx.load_audio` (already a hard dependency in every container
    that reaches this code — every backend's own `transcribe_and_diarize`
    calls it) rather than adding a new dependency (ffmpeg/soundfile don't
    reliably handle every container format Nextcloud users upload, e.g.
    m4a, without extra system packages); the redundant second decode is
    cheap next to actual transcription time for the short recordings this
    pipeline handles.

    Args:
        audio_path: Local path to the uploaded audio/video file.

    Returns:
        Duration in seconds.
    """
    import whisperx

    return len(whisperx.load_audio(audio_path)) / 16000


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
        if backend in BACKEND_REGISTRY and backend in LOCAL_BACKENDS:
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
            audio_duration = await asyncio.to_thread(_audio_duration_seconds, file.filename)
            os.remove(file.filename)
            answer = filter_transcription_chunks(chunks, max_words_per_second=wps_limit, top_n_languages=lang_limit)
            return {
                "answer": answer,
                "backend": backend,
                "removed_chunks": len(chunks) - len(answer),
                "audio_duration": audio_duration,
            }

        elif backend == "whisper" and "whisper" in LOCAL_BACKENDS:
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
            audio_duration = await asyncio.to_thread(_audio_duration_seconds, file.filename)
            os.remove(file.filename)
            answer = filter_transcription_chunks(chunks, max_words_per_second=wps_limit, top_n_languages=lang_limit)
            return {
                "answer": answer,
                "backend": backend,
                "removed_chunks": len(chunks) - len(answer),
                "audio_duration": audio_duration,
            }

        elif backend in PROXY_URLS:
            proxy_params = {
                "model_to_use": model_to_use,
                "backend": backend,
                "align": str(align),
                "include_fillers": str(include_fillers),
            }
            if num_speakers is not None:
                proxy_params["num_speakers"] = num_speakers
            if min_speakers is not None:
                proxy_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                proxy_params["max_speakers"] = max_speakers
            if max_words_per_second is not None:
                proxy_params["max_words_per_second"] = max_words_per_second
            if top_n_languages is not None:
                proxy_params["top_n_languages"] = top_n_languages

            result = await _proxy_transcribe_and_diarize(PROXY_URLS[backend], file.filename, proxy_params)
            os.remove(file.filename)
            return result

        else:
            os.remove(file.filename)
            raise HTTPException(
                status_code=404,
                detail=f"Backend '{backend}' is not available on this container and no proxy is configured for it.",
            )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/buffer_status/")
async def get_buffer_status(api_key: str = Depends(verify_api_key)):
    """Get current buffer status for debugging. Only reports backends this
    container actually runs — see LOCAL_BACKENDS."""
    status = {}
    if "whisper" in LOCAL_BACKENDS:
        status["whisper"] = whisper_buffer.get_status()
        status["diarization"] = diarization_buffer.get_status()
    status.update(
        {name.replace("-", "_"): buf.get_status() for name, buf in BACKEND_REGISTRY.items() if name in LOCAL_BACKENDS}
    )
    status["proxied_backends"] = PROXY_URLS
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
        statuses = {}
        if "whisper" in LOCAL_BACKENDS:
            statuses["whisper"] = whisper_buffer.get_status()
            statuses["diarization"] = diarization_buffer.get_status()
        statuses.update(
            {
                name.replace("-", "_"): buf.get_status()
                for name, buf in BACKEND_REGISTRY.items()
                if name in LOCAL_BACKENDS
            }
        )

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
