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

Verified against a real deploy (2026-08-21), two fixes from the initial
implementation:
1. The audio content dict's payload key is `"array"` (not `"audio"` —
   unlike GraniteSpeechBuffer's processor, which does accept `"audio"`;
   the two models' processors expect different schemas despite the similar
   chat-template shape).
2. The processor's audio feature tensors come out float32 regardless of
   the model's load dtype — `inputs.to(device)` alone left them float32
   while the model's conv layers are bfloat16, crashing with `RuntimeError:
   Input type (float) and bias type (c10::BFloat16) should be the same`.
   Fixed by `inputs.to(device, dtype=self.dtype)`, which (per
   `BatchFeature.to()`'s standard behavior) casts only floating-point
   tensors, leaving `input_ids` untouched.
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
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def load_model(self, timeout: int = 60):
        """Load ARK-ASR-3B + its processor with automatic unloading after timeout."""
        if self.is_loaded():
            self.reset_timer(timeout)
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        super().load_model(timeout=timeout)

        def _load():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, trust_remote_code=True, dtype=self.dtype, device_map=self.device
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
                    {"type": "audio", "array": chunk, "sampling_rate": 16000},
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
        ).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
