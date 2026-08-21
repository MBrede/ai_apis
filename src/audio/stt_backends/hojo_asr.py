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
"""

from datetime import datetime

from .base import DiarizeFirstASRBuffer


class HojoASRBuffer(DiarizeFirstASRBuffer):
    MODEL_ID = "HojoAI/Hojo-ASR-V1"
    DEFAULT_LANGUAGE = "Chinese"

    def load_model(self, timeout: int = 300):
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
