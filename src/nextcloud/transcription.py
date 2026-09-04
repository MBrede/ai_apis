"""
Whisper transcription: diarization params, the POST call itself, output
formatting (SRT/plain text), and the truncation ("Obacht") heuristic.
"""

import datetime
import json
import logging
import os
import re
import warnings
from pathlib import Path

import aiohttp

from src.core.auth import build_auth_headers

logger = logging.getLogger(__name__)

QWEN_MODEL_PREFIX = "qwen3-asr"
ARK_MODEL_PREFIX = "ark-asr"
HOJO_MODEL_PREFIX = "hojo-asr"
GRANITE_MODEL_PREFIX = "granite-speech"
NEMOTRON_MODEL_PREFIX = "nemotron"


def _seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_as_srt(segments: list[dict]) -> str:
    """Format diarization segments as SRT subtitles."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start_ts = _seconds_to_srt_timestamp(seg["START"])
        end_ts = _seconds_to_srt_timestamp(seg["START"] + seg["DURATION"])
        blocks.append(
            f"{i}\n{start_ts} --> {end_ts}\n{seg['SPEAKER']}: {seg['TRANSCRIPTION'].strip()}"
        )
    return "\n\n".join(blocks)


def _format_as_text(segments: list[dict]) -> str:
    """Format diarization segments as a readable plain-text transcript."""
    lines = []
    for seg in segments:
        start_str = str(datetime.timedelta(seconds=int(seg["START"])))
        end_str = str(datetime.timedelta(seconds=int(seg["START"] + seg["DURATION"])))
        lines.append(f"[{start_str} - {end_str}] {seg['SPEAKER']}: {seg['TRANSCRIPTION'].strip()}")
    return "\n".join(lines)


def _backend_for_stt_model(model: str) -> str:
    """Map a sidecar-requested STT model name to a whisper API `backend` value.

    Args:
        model: Model name from a sidecar's `stt:` field (e.g. "turbo",
            "qwen3-asr-1.7b").

    Returns:
        "qwen3-asr"/"ark-asr"/"hojo-asr"/"granite-speech"/"nemotron-asr" if
        the model name starts with the matching prefix, else "whisperx"
        (today's implicit default backend).
    """
    name = model.lower()
    if name.startswith(QWEN_MODEL_PREFIX):
        return "qwen3-asr"
    if name.startswith(ARK_MODEL_PREFIX):
        return "ark-asr"
    if name.startswith(HOJO_MODEL_PREFIX):
        return "hojo-asr"
    if name.startswith(GRANITE_MODEL_PREFIX):
        return "granite-speech"
    if name.startswith(NEMOTRON_MODEL_PREFIX):
        return "nemotron-asr"
    return "whisperx"


def _fillers_from_filename(stem: str) -> bool:
    """Return True if the filename stem requests filler-word retention.

    Recognised patterns (case-insensitive): ``_filler`` or ``_fillers`` anywhere in the stem.

    Examples::

        interview_fillers_2.mp3  → True
        meeting_filler.wav       → True
        lecture_2.mp3            → False
    """
    return bool(re.search(r"(?:^|_)fillers?(?:_|$)", stem, re.IGNORECASE))


def _speakers_from_filename(stem: str) -> int | None:
    """Try to extract a speaker count encoded in a filename stem.

    Recognised patterns (case-insensitive, anywhere in the stem):
        interview_2         → trailing _<digits>
        session_2spk        → _<digits>spk[s]
        recording_2speaker  → _<digits>speaker[s]

    Args:
        stem: Filename without extension, e.g. "interview_2" or "session_2spk".

    Returns:
        Parsed speaker count, or None if no pattern matches.
    """
    patterns = [
        r"_(\d+)speakers?$",
        r"_(\d+)spks?$",
        r"_(\d+)$",
    ]
    for pat in patterns:
        m = re.search(pat, stem, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _build_diarize_params(
    stem: str | None = None, model_override: str | None = None
) -> dict[str, str | int | bool]:
    """Build query parameters for the /transcribe_and_diarize/ endpoint.

    Speaker count priority:
      1. Encoded in the filename stem (e.g. ``interview_2.mp3`` → 2 speakers)
      2. ``NUM_SPEAKERS`` environment variable
      3. ``MIN_SPEAKERS`` + ``MAX_SPEAKERS`` environment variables

    Filler-word retention is enabled when the filename stem contains ``_filler``
    or ``_fillers`` (e.g. ``interview_fillers_2.mp3``).

    Args:
        stem: Filename stem of the file being transcribed (without extension).
        model_override: STT model requested via a sidecar's `stt:` field.
            When set, also selects the matching `backend` (see
            `_backend_for_stt_model`) instead of the endpoint's default.
            `None` preserves today's exact behavior (WHISPER_MODEL env
            default, implicit `whisperx` backend).

    Returns:
        Query-parameter dict for the diarization endpoint.

    Raises:
        ValueError: If no speaker count can be determined.
    """
    params: dict[str, str | int | bool] = {
        "model_to_use": model_override or os.environ.get("WHISPER_MODEL", "turbo"),
    }
    if model_override is not None:
        params["backend"] = _backend_for_stt_model(model_override)

    # Filler-word retention from filename
    if stem is not None and _fillers_from_filename(stem):
        logger.info("Filler-word retention enabled from filename '%s'.", stem)
        params["include_fillers"] = True

    # 1. Filename-encoded speaker count
    if stem is not None:
        n = _speakers_from_filename(stem)
        if n is not None:
            logger.info("Using %d speaker(s) from filename '%s'.", n, stem)
            params["num_speakers"] = n
            return params

    # 2. Environment variables
    num = os.environ.get("NUM_SPEAKERS")
    min_s = os.environ.get("MIN_SPEAKERS")
    max_s = os.environ.get("MAX_SPEAKERS")
    if num:
        params["num_speakers"] = int(num)
    elif min_s and max_s:
        params["min_speakers"] = int(min_s)
        params["max_speakers"] = int(max_s)
    else:
        raise ValueError(
            "Cannot determine speaker count for this file. "
            "Encode it in the filename (e.g. interview_2.mp3) or set "
            "NUM_SPEAKERS / MIN_SPEAKERS + MAX_SPEAKERS in your environment."
        )
    return params


async def _transcribe_file(
    local_path: str,
    mime_type: str,
    params: dict[str, str | int],
    session: aiohttp.ClientSession,
) -> tuple[list[dict], float | None] | None:
    """POST a file to the Whisper diarization endpoint.

    Args:
        local_path: Path to the downloaded audio/video file.
        mime_type: MIME type reported by Nextcloud.
        params: Query parameters (speaker count, model, api_key).
        session: Shared aiohttp session.

    Returns:
        `(segments, audio_duration)` on success — `audio_duration` in
        seconds, `None` if the whisper API response predates this field
        (older deployed image) rather than a hard failure. `None` (not a
        tuple) on request failure.
    """
    whisper_url = os.environ.get("WHISPER_URL", "http://whisper:8080")
    endpoint = f"{whisper_url.rstrip('/')}/transcribe_and_diarize/"
    filename = Path(local_path).name

    try:
        with open(local_path, "rb") as fh:
            form = aiohttp.FormData()
            form.add_field("file", fh, filename=filename, content_type=mime_type)
            auth_headers = build_auth_headers(os.environ.get("WHISPER_API_KEY"))
            async with session.post(
                endpoint, params=params, data=form, headers=auth_headers
            ) as resp:
                if not (200 <= resp.status < 400):
                    body = await resp.text()
                    warnings.warn(
                        f"Transcription failed for {filename} (HTTP {resp.status}): {body}"
                    )
                    return None
                body = json.loads(await resp.read())
                return body["answer"], body.get("audio_duration")
    except Exception as exc:
        warnings.warn(f"Error transcribing {filename}: {exc}")
        return None


# A transcript is only flagged if it falls short by BOTH an absolute margin
# and a proportional one — small trailing silence (someone leaving the mic
# on, applause) shouldn't trip this; a backend cutting off well before the
# real end of the file should.
TRUNCATION_MIN_GAP_SECONDS = 5.0
TRUNCATION_MIN_GAP_FRACTION = 0.1

# Max whisper transcription calls in flight at once, PER target container
# (see "container_group" below — the primary container and each of its
# qwen3-asr/hojo-asr sidecars get their own independent budget, since they're
# separate GPU-scaled deployments and shouldn't compete for the same slots).
# Lower than sync_llm.py's LLM_CONCURRENCY (default 10) on purpose — this
# hits actual GPU inference directly (no request-queuing serving layer like
# vLLM in front of it), so too much concurrency risks CUDA OOM rather than
# just queuing. Env-tunable per cluster GPU headroom.
WHISPER_CONCURRENCY = int(os.environ.get("WHISPER_CONCURRENCY", "3"))


def _detect_possible_truncation(segments: list[dict], audio_duration: float | None) -> str | None:
    """Flag a transcript whose last segment ends well before the real audio ends.

    A cheap, blunt signal for backend-side truncation (e.g. a generation-length
    cap cutting a chunk off early, or a diarized speaker turn fed to a model
    far outside its validated input length) — not a precise content check,
    and it can't catch truncation mid-file (only at the very end of the
    transcribed output). Found live: qwen3-asr's transformers-backend
    `max_new_tokens` defaulted to 512, badly truncating anything beyond a
    short clip (see `stt_backends/qwen3_asr.py`).

    Args:
        segments: This file's merged speaker segments (may be empty) — the
            same list `_format_as_text`/`_format_as_srt` render, with
            `"START"`/`"DURATION"` keys per `merge_speaker_segments`.
        audio_duration: Real audio duration in seconds, as reported by the
            whisper API (see `_audio_duration_seconds` in whisper_api.py),
            or `None` if the response predates that field (older deployed
            image) — no check is possible then, not treated as a failure.

    Returns:
        A human-readable "Obacht" message if truncation looks likely, else `None`.
    """
    if not audio_duration:
        return None
    last_end = max((s["START"] + s["DURATION"] for s in segments), default=0.0)
    gap = audio_duration - last_end
    if gap > TRUNCATION_MIN_GAP_SECONDS and gap > audio_duration * TRUNCATION_MIN_GAP_FRACTION:
        return (
            f"Obacht: dieses Transkript ist evtl. abgeschnitten — die letzte Zeile endet bei "
            f"{last_end:.1f}s, aber die Audiodatei ist {audio_duration:.1f}s lang "
            f"({gap:.1f}s / {gap / audio_duration:.0%} fehlen am Ende)."
        )
    return None
