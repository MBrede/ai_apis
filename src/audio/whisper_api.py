"""
Whisper Audio Transcription API with speaker diarization.

To start:
    gunicorn whisper_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 -t 30000
"""

import logging
import os
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from datetime import datetime

import torch
import whisper
from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from src.core.app_factory import create_app
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline

from src.core.auth_dependencies import verify_api_key
from src.core.buffer_class import Model_Buffer
from src.core.config import config

logger = logging.getLogger(__name__)


class WhisperBuffer(Model_Buffer):
    """Buffer for Whisper transcription model with automatic unloading."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None

    def load_model(self, model_name: str, timeout: int = 300, **kwargs):
        """Load Whisper model with automatic unloading after timeout."""
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        super().load_model(timeout=timeout)

        self.model = whisper.load_model(model_name, **kwargs)
        self.model_name = model_name
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """Transcribe audio file."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.reset_timer()
        return self.model.transcribe(audio_path, **kwargs)


class DiarizationBuffer(Model_Buffer):
    """Buffer for pyannote speaker diarization pipeline."""

    def __init__(self):
        super().__init__()

    def load_model(self, timeout: int = 300, **kwargs):
        """Load diarization pipeline with automatic unloading after timeout."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        super().load_model(timeout=timeout)

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=config.HF_TOKEN
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.loaded_at = datetime.now()

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

        self.reset_timer()

        if num_speakers is not None:
            return self.pipeline(audio_path, num_speakers=num_speakers)
        elif min_speakers is not None:
            return self.pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        else:
            raise ValueError(
                "Either num_speakers or min_speakers and max_speakers must be provided."
            )


class WhisperXBuffer(Model_Buffer):
    """Buffer for WhisperX pipeline: batched transcription + forced alignment + diarization."""

    def __init__(self):
        super().__init__()
        self.model_name: str = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type: str = "float16" if torch.cuda.is_available() else "int8"
        self._align_cache: dict = {}  # language_code -> (align_model, metadata)
        self._diarize_pipeline = None

    def load_model(self, model_name: str, timeout: int = 300):
        """Load WhisperX transcription model with automatic unloading after timeout."""
        if self.is_loaded() and self.model_name == model_name:
            self.reset_timer(timeout)
            return

        import whisperx

        super().load_model(timeout=timeout)

        self.model = whisperx.load_model(model_name, self.device, compute_type=self.compute_type)
        self.model_name = model_name
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def _align_model(self, language_code: str):
        import whisperx

        if language_code not in self._align_cache:
            self._align_cache[language_code] = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self._align_cache[language_code]

    def _diarize_pipeline_instance(self):
        from whisperx.diarize import DiarizationPipeline

        if self._diarize_pipeline is None:
            self._diarize_pipeline = DiarizationPipeline(
                token=config.HF_TOKEN, device=self.device
            )
        return self._diarize_pipeline

    def transcribe_and_diarize(
        self,
        audio_path: str,
        batch_size: int = 16,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        align: bool = True,
        initial_prompt: str | None = None,
    ) -> list[dict]:
        """Transcribe, align, and diarize in one pass. Speaker constraints are optional.

        Args:
            align: Run forced phoneme-level alignment before speaker assignment.
                Improves speaker boundary precision but can degrade quality when
                the wav2vec2 model has weak support for the detected language.
                Set to False to use segment-level timestamps instead.
            initial_prompt: Seed text injected via FasterWhisperPipeline.options.
                Used to nudge Whisper towards retaining filler words.
        """
        import whisperx

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.reset_timer()

        audio = whisperx.load_audio(audio_path)

        # 1. Batched transcription of full audio (preserves context, far less hallucination)
        with _patched_initial_prompt(self.model, initial_prompt):
            result = self.model.transcribe(audio, batch_size=batch_size)
        language = result["language"]

        # 2. Forced phoneme-level alignment for word timestamps (optional)
        if align:
            try:
                model_a, metadata = self._align_model(language)
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, self.device,
                    return_char_alignments=False,
                )
                logger.info("Alignment complete for language '%s'", language)
            except Exception as exc:
                logger.warning(
                    "Alignment failed for language '%s' (%s) — falling back to segment timestamps",
                    language, exc,
                )

        # 3. Diarize (speaker constraints optional — pyannote can auto-detect)
        diarize_kwargs: dict = {}
        if num_speakers is not None:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers
        elif min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = self._diarize_pipeline_instance()(audio, **diarize_kwargs)

        # 4. Assign speakers to transcript segments
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Merge consecutive segments from the same speaker into one turn
        return _merge_speaker_segments(result["segments"], language)


def _merge_speaker_segments(segments: list[dict], language: str) -> list[dict]:
    """Collapse consecutive WhisperX segments from the same speaker into one turn."""
    if not segments:
        return []

    merged: list[dict] = []
    cur = dict(segments[0])

    for seg in segments[1:]:
        if seg.get("speaker", "UNKNOWN") == cur.get("speaker", "UNKNOWN"):
            cur["end"] = seg["end"]
            cur["text"] = cur["text"].rstrip() + " " + seg["text"].strip()
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)

    return [
        {
            "SPEAKER": s.get("speaker", "UNKNOWN"),
            "START": s["start"],
            "DURATION": s["end"] - s["start"],
            "TRANSCRIPTION": s["text"].strip(),
            "LANGUAGE": language,
        }
        for s in merged
    ]


# Global buffer instances — models load on first request (lazy loading)
whisper_buffer = WhisperBuffer()
diarization_buffer = DiarizationBuffer()
whisperx_buffer = WhisperXBuffer()


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


FILLER_PROMPT = (
    "Äh, ähm, also, ja, genau, hmm, nein, okay, ne, eben, halt, eigentlich, "
    "um, uh, yeah, so, like, I mean, you know, erm, ah, right, well, actually"
)


@contextmanager
def _patched_initial_prompt(pipeline, prompt: str | None):
    """Temporarily inject initial_prompt into a FasterWhisperPipeline via its options dataclass."""
    if not prompt or not hasattr(pipeline, "options"):
        yield
        return
    original = pipeline.options
    try:
        pipeline.options = dataclass_replace(original, initial_prompt=prompt)
        yield
    finally:
        pipeline.options = original


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
        whisper_buffer.load_model(model_to_use)

    with open(file.filename, "wb") as f:
        file_contents = await file.read()
        f.write(file_contents)

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
            "whisper" uses the legacy per-segment approach (pyannote → ffmpeg → whisper).
        max_words_per_second: Remove chunks whose word rate exceeds this value.
            Human speech peaks at ~6 WPS; higher values are Whisper hallucinations.
            Set to 0 to disable.
        top_n_languages: Keep only the N most common detected languages across all
            chunks. Set to 0 to disable.
        num_speakers: (legacy backend only) exact speaker count.
            WhisperX backend accepts this as a hint but can auto-detect.
        include_fillers: Inject a prompt nudging Whisper to retain filler words
            (ähm, uhm, erm, etc.) that it would otherwise suppress.
            Only effective with the whisperx backend.
    """
    wps_limit = max_words_per_second if max_words_per_second and max_words_per_second > 0 else None
    lang_limit = top_n_languages if top_n_languages and top_n_languages > 0 else None
    filler_prompt = FILLER_PROMPT if include_fillers else None

    with open(file.filename, "wb") as f:
        file_contents = await file.read()
        f.write(file_contents)

    try:
        if backend == "whisperx":
            if not whisperx_buffer.is_loaded() or whisperx_buffer.model_name != model_to_use:
                logger.info(f"Loading WhisperX model on request: {model_to_use}")
                whisperx_buffer.load_model(model_to_use)

            chunks = whisperx_buffer.transcribe_and_diarize(
                file.filename,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                align=align,
                initial_prompt=filler_prompt,
            )
        else:
            if not whisper_buffer.is_loaded() or whisper_buffer.model_name != model_to_use:
                logger.info(f"Loading Whisper model on request: {model_to_use}")
                whisper_buffer.load_model(model_to_use)

            if not diarization_buffer.is_loaded():
                logger.info("Loading diarization pipeline on request")
                diarization_buffer.load_model()

            chunks = diarize_audio(
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
    return {
        "whisper": whisper_buffer.get_status(),
        "diarization": diarization_buffer.get_status(),
        "whisperx": whisperx_buffer.get_status(),
    }


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
        whisper_status = whisper_buffer.get_status()
        diarization_status = diarization_buffer.get_status()
        whisperx_status = whisperx_buffer.get_status()

        whisper_healthy = whisper_status is not None
        diarization_healthy = diarization_status is not None
        whisperx_healthy = whisperx_status is not None
        is_healthy = whisper_healthy and diarization_healthy and whisperx_healthy

        response_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "whisper-api",
            "whisper_buffer_accessible": whisper_healthy,
            "diarization_buffer_accessible": diarization_healthy,
            "whisperx_buffer_accessible": whisperx_healthy,
            "whisper_model_loaded": whisper_status.get("is_loaded", False) if whisper_status else False,
            "diarization_model_loaded": diarization_status.get("is_loaded", False) if diarization_status else False,
            "whisperx_model_loaded": whisperx_status.get("is_loaded", False) if whisperx_status else False,
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
