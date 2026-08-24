"""
Shared base classes and helpers for STT backends.

Every backend needs a diarization pipeline instance and GPU-safe model
loading; most backends also fit one of two shapes:

- `AlignThenDiarizeASRBuffer`: transcribe chunks -> word-level alignment ->
  diarize -> assign speakers to words. Used when the model produces (or can
  be made to produce) word-level timestamps (Qwen3-ASR, Granite-Speech).
- `DiarizeFirstASRBuffer`: diarize the whole file first (pyannote speaker
  turns), then transcribe each turn's audio slice separately. Used when the
  model has no word-level timestamps of its own (ARK-ASR-3B, Hojo-ASR-V1,
  Nemotron-3.5-ASR) — see `_diarize_then_transcribe`.

WhisperXBuffer doesn't fit either template (it uses whisperx's own batched
align API, not a manual chunk loop) — it only inherits `DiarizingASRBuffer`
for the shared diarization-pipeline/device/GPU-cleanup plumbing.
"""

import logging
from datetime import datetime

import torch

from src.core.buffer_class import Model_Buffer
from src.core.config import config

logger = logging.getLogger(__name__)


def _diarize_then_transcribe(
    diarize_pipeline,
    audio,
    transcribe_segment,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    sample_rate: int = 16000,
) -> list[dict]:
    """Diarize first, then run `transcribe_segment(audio_slice) -> str` once per
    pyannote speaker turn.

    Used by ASR backends with no word-level timestamps of their own
    (ARK-ASR-3B, Hojo-ASR-V1, Nemotron-3.5-ASR) — gives real per-turn
    speaker labels without needing word-level alignment, unlike
    WhisperX/Qwen3-ASR/Granite-Speech's transcribe-then-align-then-diarize
    path. Also sidesteps any per-call audio-length limit those models have
    (e.g. ARK-ASR-3B's 30s cap), since diarize turns are naturally short.
    """
    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["min_speakers"] = num_speakers
        diarize_kwargs["max_speakers"] = num_speakers
    elif min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
        diarize_kwargs["max_speakers"] = max_speakers
    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)

    segments = []
    for row in diarize_segments.itertuples():
        start, end = float(row.start), float(row.end)
        chunk = audio[int(start * sample_rate) : int(end * sample_rate)]
        if len(chunk) == 0:
            continue
        text = transcribe_segment(chunk)
        if text.strip():
            segments.append({"start": start, "end": end, "text": text, "speaker": row.speaker})
    return segments


def merge_speaker_segments(segments: list[dict], language: str) -> list[dict]:
    """Collapse consecutive segments from the same speaker into one turn."""
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


class DiarizingASRBuffer(Model_Buffer):
    """Base for any STT buffer that also runs pyannote diarization.

    Provides: CUDA device selection, a lazily-instantiated shared
    `DiarizationPipeline`, GPU-safe load wrapping (`_load_or_cleanup`), and
    a uniform `ensure_loaded()` entry point the API endpoint can call
    without knowing each backend's specific `load_model()` signature.
    """

    MODEL_ID: str = ""

    def __init__(self):
        super().__init__()
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._diarize_pipeline = None

    def _diarize_pipeline_instance(self):
        from whisperx.diarize import DiarizationPipeline

        if self._diarize_pipeline is None:
            try:
                self._diarize_pipeline = DiarizationPipeline(token=config.HF_TOKEN, device=self.device)
            except TypeError:
                # whisperx<3.5 (pinned for the hojo-asr container — see
                # pyproject.toml's whisper-hojoasr-only extra) named this
                # kwarg use_auth_token instead of token, AND defaults to
                # "pyannote/speaker-diarization-3.1" instead of whatever
                # newer whisperx defaults to — that model isn't gate-accepted
                # for this deployment's HF token (Pipeline.from_pretrained
                # silently returns None on gating failure, not an exception).
                # Pin explicitly to the model this deployment's token IS
                # accepted for (same one legacy DiarizationBuffer uses).
                self._diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    use_auth_token=config.HF_TOKEN,
                    device=self.device,
                )
        return self._diarize_pipeline

    def _load_or_cleanup(self, load_fn) -> None:
        """Run `load_fn()`; on failure, null out model/processor and empty the CUDA cache before re-raising.

        A failed `from_pretrained()`/`pipeline()` call (e.g. CUDA OOM
        partway through loading shards) can leave partially-placed weights
        on the GPU that nothing references anymore — `self.model` was never
        assigned, so nothing in this class holds them, but PyTorch's CUDA
        allocator doesn't know that until told. Without this, one failed
        load permanently eats GPU memory for the rest of the pod's life
        (observed: ~41GB orphaned this way for Qwen3-ASR, blocking every
        later load attempt until the pod was restarted).
        """
        try:
            load_fn()
        except Exception:
            self.model = None
            if hasattr(self, "processor"):
                self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def ensure_loaded(self, model_to_use: str | None = None) -> None:
        """Load the model if not already loaded. `model_to_use` is ignored by
        default — most backends here have exactly one size in play; only
        WhisperXBuffer overrides this to honor a requested Whisper size.
        """
        if not self.is_loaded():
            logger.info("Loading %s model on request: %s", type(self).__name__, self.MODEL_ID)
            self.load_model()


class AlignThenDiarizeASRBuffer(DiarizingASRBuffer):
    """Transcribe in chunks with word-level timestamps, then diarize and
    assign speakers to words. Used by backends that produce (or can be made
    to produce) word-level timestamps: Qwen3-ASR, Granite-Speech.

    Subclasses set `MAX_CHUNK_SECONDS` and `DEFAULT_LANGUAGE`, and implement
    `_transcribe_chunk_with_words(chunk, offset, align) -> dict` returning
    `{"start", "end", "text"}` plus an optional `"words"` list of
    `{"word", "start", "end"}` dicts (omit `"words"` to degrade to a single
    coarse chunk-level segment).
    """

    MAX_CHUNK_SECONDS: int = 300
    DEFAULT_LANGUAGE: str = "German"

    def _chunk_audio(self, audio, sample_rate: int = 16000) -> list:
        """Split a whisperx-loaded mono float32 audio array into <=MAX_CHUNK_SECONDS windows."""
        chunk_len = self.MAX_CHUNK_SECONDS * sample_rate
        return [audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len)]

    def _transcribe_chunk_with_words(self, chunk, offset: float, align: bool) -> dict:
        raise NotImplementedError

    def transcribe_and_diarize(
        self,
        audio_path: str,
        batch_size: int = 16,  # unused, kept for signature parity with WhisperXBuffer
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        align: bool = True,
        initial_prompt: str | None = None,  # unused — Whisper-specific
    ) -> list[dict]:
        """Transcribe, align, and diarize in one pass. Mirrors WhisperXBuffer's signature."""
        import whisperx

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.reset_timer()
        audio = whisperx.load_audio(audio_path)
        chunks = self._chunk_audio(audio)

        all_segments = [
            self._transcribe_chunk_with_words(chunk, i * self.MAX_CHUNK_SECONDS, align)
            for i, chunk in enumerate(chunks)
        ]
        result = {"segments": all_segments}

        diarize_kwargs: dict = {}
        if num_speakers is not None:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers
        elif min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = self._diarize_pipeline_instance()(audio, **diarize_kwargs)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        return merge_speaker_segments(result["segments"], language=self.DEFAULT_LANGUAGE)


class DiarizeFirstASRBuffer(DiarizingASRBuffer):
    """Diarize first, then transcribe each speaker turn separately. Used by
    backends with no word-level timestamps of their own: ARK-ASR-3B,
    Hojo-ASR-V1, Nemotron-3.5-ASR.

    Subclasses set `DEFAULT_LANGUAGE` and implement `_transcribe_chunk(chunk) -> str`.
    """

    DEFAULT_LANGUAGE: str = "English"

    def _transcribe_chunk(self, chunk) -> str:
        raise NotImplementedError

    def transcribe_and_diarize(
        self,
        audio_path: str,
        batch_size: int = 16,  # unused, kept for signature parity with WhisperXBuffer
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        align: bool = True,  # unused — this backend never produces word-level alignment
        initial_prompt: str | None = None,  # unused — Whisper-specific
    ) -> list[dict]:
        """Diarize, then transcribe each speaker turn. Mirrors WhisperXBuffer's signature."""
        import whisperx

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.reset_timer()
        audio = whisperx.load_audio(audio_path)
        segments = _diarize_then_transcribe(
            self._diarize_pipeline_instance(),
            audio,
            self._transcribe_chunk,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return merge_speaker_segments(segments, language=self.DEFAULT_LANGUAGE)
