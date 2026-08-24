"""
HojoAI/Hojo-ASR-V1 backend (`backend="hojo-asr"`).
https://huggingface.co/HojoAI/Hojo-ASR-V1

Languages: Mandarin, English, Cantonese, Sichuan dialect — NOT German, this
deployment's primary transcript language. Added per explicit request for
non-German/English audio; do not wire this up as a German default anywhere.

Uses the dedicated `hojo-asr` PyPI package (`pip install -U hojo-asr`), per
the model card (checked 2026-08-20) — `HOJO_ASR.load_model(...)` +
`model.run_infer(wav_paths)`, file-path-based (no in-memory-array API
documented), so `_transcribe_chunk` writes each diarize-turn slice to a
temp wav first. No word-level timestamps — `DiarizeFirstASRBuffer`, same as
ArkASRBuffer.

Dependency note: the `hojo-asr` PyPI package pins `torch<2.6`, incompatible
with this project's `torch>=2.6.0` — installed with `--no-deps` in
`Dockerfile.whisper`/`Dockerfile.whisper.hub` instead of via `pyproject.toml`.

KNOWN BROKEN as of 2026-08-24, confirmed against a real deploy. Originally
crashed with an old-transformers incompatibility (see git history) —
FIXED by splitting this backend into its own container
(ai-apis-whisper-hojoasr, see Dockerfile.whisper-hojoasr /
whisper-hojoasr-only in pyproject.toml), which resolved that specific
issue. The container is now structurally broken a different way instead:

hojo-asr's own declared dependency (torch>=2.5.1,<2.6) forces
pyannote-audio down to the 3.x line (4.x needs torch>=2.8.0 — see
whisper-hojoasr-only's comments). But this deployment's diarization model
(`pyannote/speaker-diarization-community-1`, the same one every other
backend's `_diarize_pipeline_instance()` in base.py uses successfully) has
a pipeline config that needs pyannote-audio>=4.0 (`SpeakerDiarization.
__init__() got an unexpected keyword argument 'plda'` — a clustering
option only pyannote-audio 4.x's SpeakerDiarization class accepts).
whisperx's own `DiarizationPipeline` wrapper also changed its constructor's
kwarg name between versions (`token` vs `use_auth_token` — see base.py's
try/except) and its default model changed too (whisperx<3.5 defaults to
"pyannote/speaker-diarization-3.1", not gate-accepted for this deployment's
HF token — Pipeline.from_pretrained() returns None on a gating failure
instead of raising, so base.py's fallback pins model_name explicitly to
community-1 instead of relying on that default).

Net result: no pyannote-audio version can satisfy both "compatible with
hojo-asr's torch<2.6" and "can load this deployment's diarization model."
Not fixable from this file — would need hojo-asr itself to relax its torch
pin upstream, or a pyannote-audio 3.x-compatible diarization model this
deployment's HF token has gated access to. Given this backend also doesn't
support German (this deployment's primary language), not prioritized
further.
"""

from datetime import datetime

from .base import DiarizeFirstASRBuffer


class HojoASRBuffer(DiarizeFirstASRBuffer):
    MODEL_ID = "HojoAI/Hojo-ASR-V1"
    DEFAULT_LANGUAGE = "Chinese"

    def load_model(self, timeout: int = 60):
        """Load Hojo-ASR-V1 with automatic unloading after timeout."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        from hojo_asr import HOJO_ASR

        super().load_model(timeout=timeout)

        def _load():
            self.model = HOJO_ASR.load_model(self.MODEL_ID, device=self.device)

        self._load_or_cleanup(_load)
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def _transcribe_chunk(self, chunk) -> str:
        """Write a diarize-turn slice to a temp wav and run one Hojo run_infer() call."""
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, chunk, 16000)
            result = self.model.run_infer([tmp.name])
        return result[0]["text"] if result else ""
