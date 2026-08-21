"""
Tests for sync.py's sidecar-driven multi-model STT and multi-folder helpers.
"""

from src.nextcloud.sync import (
    _backend_for_stt_model,
    _make_webdav_client,
    _normalize_list,
    _parse_sidecar_stt,
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
