"""
Tests for sync_llm.py's cartesian llm x prompt scheduling, context injection,
and multi-folder prompt-cache namespacing.
"""

from src.nextcloud.sync_llm import DEFAULT_PROMPT, _parse_sidecar, _render_prompt


class TestParseSidecar:
    """Test the unified llm/prompt/context sidecar field parsing."""

    def test_empty_sidecar_falls_back_to_single_item_defaults(self, monkeypatch):
        monkeypatch.delenv("LLM_DEFAULT_MODEL", raising=False)
        parsed = _parse_sidecar("")
        assert parsed == {"models": ["glm-4-7-flash"], "prompts": [DEFAULT_PROMPT], "context": ""}

    def test_env_default_model_used_when_llm_field_absent(self, monkeypatch):
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "qwen3-14b")
        parsed = _parse_sidecar("")
        assert parsed["models"] == ["qwen3-14b"]

    def test_scalar_llm_and_prompt_become_one_element_lists(self):
        parsed = _parse_sidecar("llm: glm-4-7-flash\nprompt: summary\n")
        assert parsed["models"] == ["glm-4-7-flash"]
        assert parsed["prompts"] == ["summary"]

    def test_list_llm_and_prompt_preserved(self):
        parsed = _parse_sidecar("llm: [glm-4-7-flash, qwen3-14b]\nprompt: [summary, action-items]\n")
        assert parsed["models"] == ["glm-4-7-flash", "qwen3-14b"]
        assert parsed["prompts"] == ["summary", "action-items"]

    def test_context_field_passed_through(self):
        parsed = _parse_sidecar("context: Weekly Benkana sync\n")
        assert parsed["context"] == "Weekly Benkana sync"

    def test_missing_context_defaults_to_empty_string(self):
        parsed = _parse_sidecar("")
        assert parsed["context"] == ""

    def test_stt_field_is_ignored_here(self):
        # stt: belongs to sync.py's whisper stage, not the LLM sidecar parse.
        parsed = _parse_sidecar("stt: [turbo, qwen3-asr-1.7b]\n")
        assert "stt" not in parsed


class TestRenderPrompt:
    """Test the {transcript}/{context} template placeholder substitution."""

    def test_transcript_placeholder_substituted(self):
        assert _render_prompt("Summarize: {transcript}", "hello world") == "Summarize: hello world"

    def test_context_placeholder_substituted(self):
        result = _render_prompt("{context}\n\n{transcript}", "the transcript", "some context")
        assert result == "some context\n\nthe transcript"

    def test_missing_context_arg_defaults_to_empty_no_op(self):
        # Template without {context} is unaffected by the new default param.
        assert _render_prompt("Summarize: {transcript}", "hello") == "Summarize: hello"

    def test_template_without_context_placeholder_ignores_context_arg(self):
        result = _render_prompt("Summarize: {transcript}", "hello", "unused context")
        assert result == "Summarize: hello"


class TestTranscriptStemMatching:
    """Test deriving expected transcript stems from a sidecar's own `stt:`
    field (mirrors the logic in _collect_llm_jobs: expected_stems computed
    from the sidecar, then intersected with what's actually on disk yet —
    NOT a folder-guessing prefix match, which is ambiguous: "interview_2" is
    a different audio file's transcript per sync.py's speaker-count filename
    convention, not an "interview" file with stt-suffix "2")."""

    @staticmethod
    def _expected_stems(stem: str, stt_models: list[str]) -> list[str]:
        return [stem] if not stt_models else [f"{stem}_{m}" for m in stt_models]

    def test_no_sidecar_stt_field_expects_bare_stem(self):
        assert self._expected_stems("interview", []) == ["interview"]

    def test_stt_list_expects_one_stem_per_model(self):
        assert self._expected_stems("interview", ["turbo", "qwen3-asr-1.7b"]) == [
            "interview_turbo",
            "interview_qwen3-asr-1.7b",
        ]

    def test_intersecting_with_existing_transcripts_ignores_unrelated_files(self):
        expected = self._expected_stems("interview", ["turbo"])
        existing = {"interview_turbo", "interview_2", "interview_2_turbo"}
        # "interview_2*" belongs to a different audio file — must not leak in,
        # even though a naive prefix match against `existing` would catch it.
        assert [s for s in expected if s in existing] == ["interview_turbo"]

    def test_partially_ready_transcripts_still_processes_whats_available(self):
        expected = self._expected_stems("interview", ["turbo", "qwen3-asr-1.7b"])
        existing = {"interview_turbo"}  # qwen3-asr-1.7b transcript not done yet
        assert [s for s in expected if s in existing] == ["interview_turbo"]


class TestCartesianJobScheduling:
    """Test the model-grouping sort used by main() (unit-testable in isolation)."""

    def test_stable_sort_groups_by_model_preserving_relative_order(self):
        jobs = [
            {"model": "b", "prompt": "p1"},
            {"model": "a", "prompt": "p1"},
            {"model": "b", "prompt": "p2"},
            {"model": "a", "prompt": "p2"},
        ]
        jobs.sort(key=lambda j: j["model"])
        assert [j["model"] for j in jobs] == ["a", "a", "b", "b"]
        # Intra-group relative order preserved (stability).
        assert [j["prompt"] for j in jobs] == ["p1", "p2", "p1", "p2"]
