"""
Shared constants for the Nextcloud sync package.

Standalone by design — no imports from sibling modules in this package, so
every other module here (sync.py, sync_llm.py, and everything under them)
can import from this one without any risk of a circular import.
"""

from pathlib import Path

AUDIO_VIDEO_MIME_TYPES: frozenset[str] = frozenset(
    [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/x-m4a",
        "audio/mp4",
        "audio/flac",
        "audio/aac",
        "audio/webm",
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "video/avi",
    ]
)

TRANSCRIPT_SUBFOLDER = "transcriptions"
LLM_SUBFOLDER = "llm"
PROMPTS_SUBFOLDER = "prompts"
JUDGES_SUBFOLDER = "judges"
LOCAL_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "summary"

# Max candidate tokens vLLM/OpenAI returns per generated-token position when
# logprobs are requested. No max_tokens cap is set on judge calls (see
# judges.py's _call_judge_llm) — a reasoning-tuned judge model emits
# <think>...</think> as a mandatory part of its generation order, so
# truncating the token budget would cut it off mid-thought before it ever
# reaches the answer.
JUDGE_TOP_LOGPROBS = 20

SIDECAR_SUFFIXES = (".yaml", ".yml")
BATCH_FILENAMES = ("batch.xlsx", "batch.csv")  # checked in this order; first match wins
BATCH_ROW_LIST_FIELDS = ("stt", "llm", "prompt")
BATCH_ROW_RESERVED_FIELDS = frozenset({"filename", "file", "context", *BATCH_ROW_LIST_FIELDS})
