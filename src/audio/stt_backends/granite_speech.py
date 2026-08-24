"""
Granite-Speech-4.1-2B-Plus backend (`backend="granite-speech"`).
https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus

Unlike ARK-ASR/Hojo-ASR/Nemotron-ASR, this model DOES produce real
word-level timestamps via a dedicated prompt (`[T:N]` tags, end time in
centiseconds with a 10s modulo rollover) — hence it's an
`AlignThenDiarizeASRBuffer`, not diarize-first. Supports German (English,
French, German, Spanish, Portuguese per the model card).

Chunked at 3.5 minutes per call — the model card's own tested limit for the
timestamp task specifically (vs. 9 minutes for plain ASR).

Implementation-time risk (same caveat style as Qwen3ASRBuffer/ArkASRBuffer):
the model card's own example wraps `processor`/`model` in an untyped
`transcribe(audio, prompt, max_new_tokens=...)` helper whose internals
aren't shown — `_transcribe_chunk_raw` below reimplements it via
`apply_chat_template` + `generate()` (same shape as ArkASRBuffer), not
verified against a real run. `[T:N]`-tag parsing only has an *end* time per
word (no start) — `start` is approximated as the previous word's `end`
(0.0 for the first word), which is what `assign_word_speakers` needs but is
coarser than WhisperX/Qwen3-ASR's true per-word spans. A
malformed/unparseable response degrades to a single coarse chunk-level
segment (no `words` key), same graceful-degradation style as
Qwen3ASRBuffer._extract_word_timestamps.
"""

import logging
import re
from datetime import datetime

import torch

from .base import AlignThenDiarizeASRBuffer

logger = logging.getLogger(__name__)


class GraniteSpeechBuffer(AlignThenDiarizeASRBuffer):
    MODEL_ID = "ibm-granite/granite-speech-4.1-2b-plus"
    MAX_CHUNK_SECONDS = 210  # 3.5 min — model card's tested limit for the timestamp task
    DEFAULT_LANGUAGE = "German"
    TS_PROMPT = (
        "<|audio|> Timestamps: Transcribe the speech. After each word, add a "
        "timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
    )

    def __init__(self):
        super().__init__()
        self.processor = None

    def load_model(self, timeout: int = 60):
        """Load Granite-Speech-4.1-2B-Plus + its processor with automatic unloading after timeout."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        super().load_model(timeout=timeout)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        def _load():
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.MODEL_ID, dtype=dtype, device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        self._load_or_cleanup(_load)
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def _transcribe_chunk_raw(self, chunk) -> str:
        """Run one generate() call with the timestamp prompt on a <=3.5min float32 16kHz slice."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": chunk, "sampling_rate": 16000},
                    {"type": "text", "text": self.TS_PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=10000)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _transcribe_chunk_with_words(self, chunk, offset: float, align: bool) -> dict:
        ts_text = self._transcribe_chunk_raw(chunk)
        words = self._extract_word_timestamps(ts_text, offset) if align else []
        plain_text = re.sub(r"\[T:\d+\]", "", ts_text).strip()
        if words:
            return {"start": words[0]["start"], "end": words[-1]["end"], "text": plain_text, "words": words}
        return {"start": offset, "end": offset + len(chunk) / 16000, "text": plain_text}

    @classmethod
    def _extract_word_timestamps(cls, ts_text: str, offset_seconds: float) -> list[dict]:
        """Parse `[T:N]` centisecond end-tags into whisperx's expected word dict shape.

        See module docstring's "Implementation-time risk" note — `start` is
        approximated from the previous word's `end` (best available given
        the tag format only carries an end time). Returns `[]` (not a
        crash) on any parse failure, degrading gracefully to a single
        coarse chunk-level segment in the caller.
        """
        try:
            parts = re.split(r"\[T:(\d+)\]", ts_text)
            words: list[dict] = []
            offset_cs = 0
            last_raw = 0
            prev_end = 0.0
            for i in range(0, len(parts) - 1, 2):
                text_chunk = parts[i].strip()
                raw = int(parts[i + 1])
                if raw < last_raw:
                    offset_cs += 1000  # 10s modulo rollover
                last_raw = raw
                end = offset_seconds + (offset_cs + raw) / 100.0
                for w in text_chunk.split():
                    words.append({"word": w, "start": prev_end, "end": end})
                    prev_end = end
            return words
        except (ValueError, IndexError) as exc:
            logger.warning("Unexpected Granite-Speech timestamp tag shape (%s) — skipping.", exc)
            return []
