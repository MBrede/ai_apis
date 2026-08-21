"""
Audio8/ARK-ASR-3B backend (`backend="ark-asr"`).
https://huggingface.co/Audio8/ARK-ASR-3B

Whisper-encoder + Qwen decoder, loaded via `transformers` with
`trust_remote_code=True` (no dedicated PyPI package, per the model card,
checked 2026-08-20). Its `generate()` API returns plain text with no
word-level timestamps, so this is a `DiarizeFirstASRBuffer` — diarize the
whole file first (pyannote speaker turns), then transcribe each turn. This
also works around the model's documented 30s-per-call audio limit for free
(diarize turns are naturally short).

Implementation-time risk (same caveat style as Qwen3ASRBuffer): the exact
`processor.apply_chat_template()` kwarg names below are inferred from the
model card's description, not a verified working run — if the shape is
off, `_transcribe_chunk` raising is the expected failure mode; check the
model card's own example script if this needs debugging.
"""

from datetime import datetime

import torch

from .base import DiarizeFirstASRBuffer


class ArkASRBuffer(DiarizeFirstASRBuffer):
    MODEL_ID = "Audio8/ARK-ASR-3B"
    DEFAULT_LANGUAGE = "German"

    def __init__(self):
        super().__init__()
        self.processor = None

    def load_model(self, timeout: int = 300):
        """Load ARK-ASR-3B + its processor with automatic unloading after timeout."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        super().load_model(timeout=timeout)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        def _load():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, trust_remote_code=True, dtype=dtype, device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)

        self._load_or_cleanup(_load)
        self.loaded_at = datetime.now()

        if self.timer:
            self.timer.start()

    def _transcribe_chunk(self, chunk) -> str:
        """Run one ARK generate() call on a <=30s float32 16kHz audio slice."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": chunk, "sampling_rate": 16000},
                    {"type": "text", "text": "Transcribe the audio."},
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
            output_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
