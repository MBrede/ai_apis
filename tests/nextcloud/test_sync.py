"""
Tests for sync.py's sidecar-driven multi-model STT and multi-folder helpers.
"""

import openpyxl

from src.nextcloud.sync import (
    _backend_for_stt_model,
    _detect_possible_truncation,
    _make_webdav_client,
    _normalize_list,
    _parse_sidecar_stt,
    _read_batch_csv,
    _read_batch_xlsx,
    _row_to_sidecar_dict,
    _stt_models_from_dict,
)


class TestNormalizeList:
    """Test the scalar/list/absent sidecar-field normalization helper."""

    def test_none_returns_empty_list(self):
        assert _normalize_list(None) == []

    def test_scalar_returns_one_item_list(self):
        assert _normalize_list("turbo") == ["turbo"]

    def test_list_returns_stringified_list(self):
        assert _normalize_list(["turbo", "qwen3-asr-1.7b"]) == ["turbo", "qwen3-asr-1.7b"]

    def test_non_string_scalar_is_stringified(self):
        assert _normalize_list(42) == ["42"]


class TestParseSidecarStt:
    """Test the `stt:` sidecar field parsing."""

    def test_empty_sidecar_returns_empty_list(self):
        assert _parse_sidecar_stt("") == []

    def test_no_stt_field_returns_empty_list(self):
        assert _parse_sidecar_stt("llm: glm-4-7-flash\n") == []

    def test_single_model_scalar(self):
        assert _parse_sidecar_stt("stt: turbo\n") == ["turbo"]

    def test_multiple_models_list(self):
        assert _parse_sidecar_stt("stt: [turbo, qwen3-asr-1.7b]\n") == ["turbo", "qwen3-asr-1.7b"]


class TestBackendForSttModel:
    """Test the sidecar STT model name -> whisper API `backend` mapping."""

    def test_whisper_model_name_maps_to_whisperx(self):
        assert _backend_for_stt_model("turbo") == "whisperx"

    def test_qwen3_asr_model_name_maps_to_qwen3_asr_backend(self):
        assert _backend_for_stt_model("qwen3-asr-1.7b") == "qwen3-asr"

    def test_mapping_is_case_insensitive(self):
        assert _backend_for_stt_model("Qwen3-ASR-1.7B") == "qwen3-asr"

    def test_ark_asr_model_name_maps_to_ark_asr_backend(self):
        assert _backend_for_stt_model("ark-asr-3b") == "ark-asr"

    def test_hojo_asr_model_name_maps_to_hojo_asr_backend(self):
        assert _backend_for_stt_model("hojo-asr-v1") == "hojo-asr"

    def test_granite_speech_model_name_maps_to_granite_speech_backend(self):
        assert _backend_for_stt_model("granite-speech-4.1-2b-plus") == "granite-speech"

    def test_nemotron_model_name_maps_to_nemotron_asr_backend(self):
        assert _backend_for_stt_model("nemotron-3.5-asr-streaming-0.6b") == "nemotron-asr"


class TestMakeWebdavClientMultiFolder:
    """Test NEXTCLOUD_FOLDER comma-separated multi-folder parsing."""

    def _set_base_env(self, monkeypatch):
        monkeypatch.setenv("NEXTCLOUD_URL", "https://example.com")
        monkeypatch.setenv("NEXTCLOUD_USER", "user")
        monkeypatch.setenv("NEXTCLOUD_PASSWORD", "pw")
        # NEXTCLOUD_DAV_USER defaults to NEXTCLOUD_USER when unset — clear it
        # explicitly so a real deployment .env picked up by pytest-dotenv
        # can't leak a different value into these tests.
        monkeypatch.delenv("NEXTCLOUD_DAV_USER", raising=False)

    def test_single_folder_produces_one_element_list(self, monkeypatch):
        self._set_base_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_FOLDER", "transcription")
        _client, roots = _make_webdav_client()
        assert roots == ["/remote.php/dav/files/user/transcription"]

    def test_multiple_folders_comma_separated(self, monkeypatch):
        self._set_base_env(monkeypatch)
        monkeypatch.setenv(
            "NEXTCLOUD_FOLDER",
            "transcription,/Shared/1 Projects/MBS Benkana/transcription",
        )
        _client, roots = _make_webdav_client()
        assert roots == [
            "/remote.php/dav/files/user/transcription",
            "/remote.php/dav/files/user/Shared/1 Projects/MBS Benkana/transcription",
        ]

    def test_extra_whitespace_and_slashes_are_stripped(self, monkeypatch):
        self._set_base_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_FOLDER", " /a/ , /b/ ")
        _client, roots = _make_webdav_client()
        assert roots == [
            "/remote.php/dav/files/user/a",
            "/remote.php/dav/files/user/b",
        ]


class TestSttModelsFromDict:
    """Test the dict-based core shared by _parse_sidecar_stt (YAML) and the batch-row path."""

    def test_empty_dict_returns_empty_list(self):
        assert _stt_models_from_dict({}) == []

    def test_scalar_stt_returns_one_item_list(self):
        assert _stt_models_from_dict({"stt": "turbo"}) == ["turbo"]

    def test_list_stt_returned_as_is(self):
        assert _stt_models_from_dict({"stt": ["turbo", "qwen3-asr-1.7b"]}) == ["turbo", "qwen3-asr-1.7b"]


class TestRowToSidecarDict:
    """Test converting one batch.csv/batch.xlsx row into the sidecar-dict shape."""

    def test_empty_row_returns_empty_dict(self):
        assert _row_to_sidecar_dict({}) == {}

    def test_single_value_fields_become_one_item_lists(self):
        row = {"stt": "turbo", "llm": "glm-4-7-flash", "prompt": "summary"}
        assert _row_to_sidecar_dict(row) == {
            "stt": ["turbo"],
            "llm": ["glm-4-7-flash"],
            "prompt": ["summary"],
        }

    def test_comma_separated_values_become_lists(self):
        row = {"stt": "turbo, qwen3-asr-1.7b", "llm": "glm-4-7-flash,qwen3-14b"}
        assert _row_to_sidecar_dict(row) == {
            "stt": ["turbo", "qwen3-asr-1.7b"],
            "llm": ["glm-4-7-flash", "qwen3-14b"],
        }

    def test_unknown_column_passed_through_as_extra_field(self):
        row = {"filename": "demo_a", "llm": "glm-4-7-flash", "ground_truth": "expected reference text"}
        result = _row_to_sidecar_dict(row)
        assert result["ground_truth"] == "expected reference text"
        assert "filename" not in result  # reserved, not a passthrough field

    def test_blank_extra_column_included_as_empty_string(self):
        # Found live: skipping a blank extra column entirely left a judge's
        # {ground_truth} placeholder literally unreplaced for files with no
        # value in that column, instead of substituting "" like {context}.
        row = {"llm": "glm-4-7-flash", "ground_truth": "   "}
        assert _row_to_sidecar_dict(row)["ground_truth"] == ""

    def test_context_passed_through_as_plain_text_not_split(self):
        row = {"context": "Case notes: patient reported dizziness, not nausea."}
        assert _row_to_sidecar_dict(row) == {
            "context": "Case notes: patient reported dizziness, not nausea."
        }

    def test_blank_and_missing_fields_omitted(self):
        row = {"stt": "", "llm": "  ", "prompt": "summary", "context": None}
        assert _row_to_sidecar_dict(row) == {"prompt": ["summary"]}


class TestReadBatchCsv:
    """Test reading a real batch.csv file end to end."""

    def test_reads_rows_with_lowercased_headers(self, tmp_path):
        csv_path = tmp_path / "batch.csv"
        csv_path.write_text(
            "Filename,STT,LLM,Prompt,Context\n"
            "interview_1.m4a,turbo,glm-4-7-flash,summary,Case A\n"
            "interview_2.m4a,\"turbo, qwen3-asr-1.7b\",glm-4-7-flash,summary,Case B\n",
            encoding="utf-8",
        )
        rows = _read_batch_csv(csv_path)
        assert rows == [
            {
                "filename": "interview_1.m4a",
                "stt": "turbo",
                "llm": "glm-4-7-flash",
                "prompt": "summary",
                "context": "Case A",
            },
            {
                "filename": "interview_2.m4a",
                "stt": "turbo, qwen3-asr-1.7b",
                "llm": "glm-4-7-flash",
                "prompt": "summary",
                "context": "Case B",
            },
        ]


class TestReadBatchXlsx:
    """Test reading a real batch.xlsx file end to end."""

    def test_reads_rows_with_lowercased_headers(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Filename", "STT", "LLM", "Prompt", "Context"])
        ws.append(["interview_1.m4a", "turbo", "glm-4-7-flash", "summary", "Case A"])
        xlsx_path = tmp_path / "batch.xlsx"
        wb.save(xlsx_path)

        rows = _read_batch_xlsx(xlsx_path)
        assert rows == [
            {
                "filename": "interview_1.m4a",
                "stt": "turbo",
                "llm": "glm-4-7-flash",
                "prompt": "summary",
                "context": "Case A",
            }
        ]

    def test_blank_trailing_row_skipped(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Filename", "Prompt"])
        ws.append(["interview_1.m4a", "summary"])
        ws.append([None, None])
        xlsx_path = tmp_path / "batch.xlsx"
        wb.save(xlsx_path)

        rows = _read_batch_xlsx(xlsx_path)
        assert len(rows) == 1


class TestDetectPossibleTruncation:
    """Test the end-of-transcript-vs-audio-duration truncation heuristic.

    Found live: qwen3-asr's transformers-backend max_new_tokens defaulted
    to 512, badly truncating anything beyond a short clip — this check is
    a cheap, blunt safety net for that class of bug, not a precise one.
    """

    def _segments(self, *ends: float) -> list[dict]:
        return [{"START": 0.0, "DURATION": end, "SPEAKER": "SPEAKER_00", "TRANSCRIPTION": "x"} for end in ends]

    def test_no_audio_duration_skips_check(self):
        assert _detect_possible_truncation(self._segments(10.0), None) is None

    def test_transcript_covers_full_duration_no_flag(self):
        assert _detect_possible_truncation(self._segments(298.0), 300.0) is None

    def test_small_trailing_gap_not_flagged(self):
        # Natural trailing silence — under both the absolute and proportional threshold.
        assert _detect_possible_truncation(self._segments(298.0), 300.0) is None
        assert _detect_possible_truncation(self._segments(3.0), 4.0) is None  # 1s/4s=25% but <5s absolute

    def test_badly_truncated_transcript_flagged(self):
        # Matches the real qwen3-asr bug shape: a 300s chunk producing only ~40s of output.
        msg = _detect_possible_truncation(self._segments(40.0), 300.0)
        assert msg is not None
        assert "Obacht" in msg

    def test_empty_segments_with_real_audio_flagged(self):
        msg = _detect_possible_truncation([], 60.0)
        assert msg is not None

    def test_requires_both_absolute_and_proportional_margin(self):
        # 8s gap on a 1000s file is proportionally tiny (0.8%) — not flagged
        # despite exceeding the absolute-seconds threshold alone.
        assert _detect_possible_truncation(self._segments(992.0), 1000.0) is None
