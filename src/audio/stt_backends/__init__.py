"""
STT backend buffers: one module per model, sharing two template base
classes (`AlignThenDiarizeASRBuffer`, `DiarizeFirstASRBuffer`) plus the
legacy Whisper+pyannote pair. See `base.py` for the class hierarchy.

Exposes ready-to-use singleton instances (models load lazily on first
request) and `BACKEND_REGISTRY`, mapping the `/transcribe_and_diarize/`
endpoint's `backend` query param to its buffer — the endpoint only needs to
special-case `"whisperx"` (honors a requested Whisper size) and the legacy
`"whisper"` fallback; every other name is a uniform `buf.ensure_loaded()` +
`buf.transcribe_and_diarize()` call.
"""

from .ark_asr import ArkASRBuffer
from .base import DiarizingASRBuffer
from .diarization import DiarizationBuffer
from .granite_speech import GraniteSpeechBuffer
from .hojo_asr import HojoASRBuffer
from .nemotron_asr import NemotronASRBuffer
from .qwen3_asr import Qwen3ASRBuffer
from .whisper import WhisperBuffer
from .whisperx_backend import FILLER_PROMPT, WhisperXBuffer

__all__ = [
    "WhisperBuffer",
    "DiarizationBuffer",
    "WhisperXBuffer",
    "Qwen3ASRBuffer",
    "GraniteSpeechBuffer",
    "ArkASRBuffer",
    "HojoASRBuffer",
    "NemotronASRBuffer",
    "FILLER_PROMPT",
    "BACKEND_REGISTRY",
    "whisper_buffer",
    "diarization_buffer",
    "whisperx_buffer",
    "qwen3_asr_buffer",
    "granite_speech_buffer",
    "ark_asr_buffer",
    "hojo_asr_buffer",
    "nemotron_asr_buffer",
]

# Global buffer instances — models load on first request (lazy loading)
whisper_buffer = WhisperBuffer()
diarization_buffer = DiarizationBuffer()
whisperx_buffer = WhisperXBuffer()
qwen3_asr_buffer = Qwen3ASRBuffer()
granite_speech_buffer = GraniteSpeechBuffer()
ark_asr_buffer = ArkASRBuffer()
hojo_asr_buffer = HojoASRBuffer()
nemotron_asr_buffer = NemotronASRBuffer()

BACKEND_REGISTRY: dict[str, DiarizingASRBuffer] = {
    "whisperx": whisperx_buffer,
    "qwen3-asr": qwen3_asr_buffer,
    "granite-speech": granite_speech_buffer,
    "ark-asr": ark_asr_buffer,
    "hojo-asr": hojo_asr_buffer,
    "nemotron-asr": nemotron_asr_buffer,
}
