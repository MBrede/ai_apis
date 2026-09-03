"""
Tests for sync_llm.py's cartesian llm x prompt scheduling, context injection,
and multi-folder prompt-cache namespacing.
"""

import pytest

from src.nextcloud.sync_llm import (
    DEFAULT_PROMPT,
    _json_schema_for_fields,
    _load_prompt_template,
    _parse_sidecar,
    _render_prompt,
)


class TestParseSidecar:
    """Test the unified llm/prompt/context sidecar field parsing."""

    def test_empty_sidecar_falls_back_to_single_item_defaults(self, monkeypatch):
        monkeypatch.delenv("LLM_DEFAULT_MODEL", raising=False)
        parsed = _parse_sidecar("")
        assert parsed == {
            "models": ["glm-4-7-flash"],
            "prompts": [DEFAULT_PROMPT],
            "context": "",
            "extra_fields": {},
        }

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

    def test_extra_field_passed_through(self):
        parsed = _parse_sidecar("ground_truth: The user was confused by the RFID term.\n")
        assert parsed["extra_fields"] == {"ground_truth": "The user was confused by the RFID term."}

    def test_known_fields_excluded_from_extra_fields(self):
        parsed = _parse_sidecar("stt: turbo\nllm: glm-4-7-flash\nprompt: summary\ncontext: hi\n")
        assert parsed["extra_fields"] == {}


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

    def test_extra_field_placeholder_substituted(self):
        result = _render_prompt(
            "Ground truth: {ground_truth}\n{transcript}", "hello", extra_fields={"ground_truth": "expected answer"}
        )
        assert result == "Ground truth: expected answer\nhello"

    def test_missing_extra_fields_arg_is_a_no_op(self):
        assert _render_prompt("Summarize: {transcript}", "hello") == "Summarize: hello"


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


class TestJsonSchemaForFields:
    """Test the response_format JSON schema built from a structured prompt's fields:."""

    def test_every_field_becomes_a_required_string_property(self):
        schema = _json_schema_for_fields({"summary": "one paragraph", "action_items": "bullet list"})
        props = schema["json_schema"]["schema"]["properties"]
        assert props == {
            "summary": {"type": "string", "description": "one paragraph"},
            "action_items": {"type": "string", "description": "bullet list"},
        }
        assert schema["json_schema"]["schema"]["required"] == ["summary", "action_items"]

    def test_response_format_type_is_json_schema(self):
        schema = _json_schema_for_fields({"x": "y"})
        assert schema["type"] == "json_schema"
        assert schema["json_schema"]["schema"]["type"] == "object"


class TestLoadPromptTemplate:
    """Test .md (plain) vs .yaml/.yml (structured) prompt loading."""

    class _FakeClient:
        def __init__(self, files):
            self._files = files
            self.download_count = 0

        def download_sync(self, remote_path, local_path):
            self.download_count += 1
            if remote_path not in self._files:
                from webdav3.exceptions import RemoteResourceNotFound

                raise RemoteResourceNotFound(remote_path)
            from pathlib import Path

            Path(local_path).write_text(self._files[remote_path], encoding="utf-8")

    def test_plain_md_prompt_has_no_fields(self, tmp_path):
        client = self._FakeClient({"root/prompts/summary.md": "Summarize: {transcript}"})
        result = _load_prompt_template(client, "root", "summary", tmp_path, {})
        assert result == {"template": "Summarize: {transcript}", "fields": None}

    def test_yaml_prompt_returns_template_and_fields(self, tmp_path):
        yaml_text = """
prompt: "Summarize: {transcript}"
fields:
  summary: one paragraph
  action_items: bullet list
"""
        client = self._FakeClient({"root/prompts/notes.yaml": yaml_text})
        result = _load_prompt_template(client, "root", "notes", tmp_path, {})
        assert result == {
            "template": "Summarize: {transcript}",
            "fields": {"summary": "one paragraph", "action_items": "bullet list"},
        }

    def test_yml_extension_also_works(self, tmp_path):
        yaml_text = "prompt: hi {transcript}\nfields:\n  x: description\n"
        client = self._FakeClient({"root/prompts/notes.yml": yaml_text})
        result = _load_prompt_template(client, "root", "notes", tmp_path, {})
        assert result["fields"] == {"x": "description"}

    def test_yaml_takes_precedence_over_md(self, tmp_path):
        client = self._FakeClient(
            {
                "root/prompts/notes.yaml": "prompt: yaml version\nfields:\n  x: y\n",
                "root/prompts/notes.md": "md version",
            }
        )
        result = _load_prompt_template(client, "root", "notes", tmp_path, {})
        assert result["template"] == "yaml version"

    def test_yaml_missing_fields_raises(self, tmp_path):
        client = self._FakeClient({"root/prompts/notes.yaml": "prompt: hi\n"})
        with pytest.raises(ValueError, match="fields"):
            _load_prompt_template(client, "root", "notes", tmp_path, {})

    def test_yaml_missing_prompt_raises(self, tmp_path):
        client = self._FakeClient({"root/prompts/notes.yaml": "fields:\n  x: y\n"})
        with pytest.raises(ValueError, match="prompt"):
            _load_prompt_template(client, "root", "notes", tmp_path, {})

    def test_yaml_empty_fields_dict_raises(self, tmp_path):
        client = self._FakeClient({"root/prompts/notes.yaml": "prompt: hi\nfields: {}\n"})
        with pytest.raises(ValueError, match="fields"):
            _load_prompt_template(client, "root", "notes", tmp_path, {})

    def test_nothing_found_raises_file_not_found(self, tmp_path):
        client = self._FakeClient({})
        with pytest.raises(FileNotFoundError):
            _load_prompt_template(client, "root", "missing", tmp_path, {})

    def test_result_is_cached_on_second_call(self, tmp_path):
        # .md is only found after probing (and missing) .yaml/.yml first, so
        # the first lookup alone makes more than one download_sync call —
        # what matters here is that the SECOND lookup makes none at all.
        client = self._FakeClient({"root/prompts/summary.md": "hi"})
        cache = {}
        _load_prompt_template(client, "root", "summary", tmp_path, cache)
        count_after_first = client.download_count
        _load_prompt_template(client, "root", "summary", tmp_path, cache)
        assert client.download_count == count_after_first
