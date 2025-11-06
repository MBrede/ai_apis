"""
Tests for the text classification API.

Tests API endpoints, model loading, and prediction functionality.
"""

from unittest.mock import Mock, patch

import pytest

# Skip these tests if the required packages aren't installed
torch = pytest.importorskip("torch")
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def mock_classification_model():
    """Mock text classification model."""
    model = Mock()
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    return model


@pytest.fixture
def mock_setfit_model():
    """Mock SetFit model."""
    model = Mock()
    model.id2label = {0: "negative", 1: "positive"}
    model.predict_proba = Mock(return_value=[[0.2, 0.8], [0.7, 0.3]])
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    return tokenizer


class TestClassificationBuffer:
    """Test ClassificationBuffer class."""

    @patch("src.text_analysis.text_classification_api.hf_api")
    @patch("src.text_analysis.text_classification_api.AutoModelForSequenceClassification")
    @patch("src.text_analysis.text_classification_api.AutoTokenizer")
    def test_load_standard_model(
        self,
        mock_tokenizer_class,
        mock_model_class,
        mock_hf_api,
        mock_classification_model,
        mock_tokenizer,
    ):
        """Test loading a standard transformer model."""
        from src.text_analysis.text_classification_api import ClassificationBuffer

        # Mock HuggingFace API response for standard model
        mock_model_info = Mock()
        mock_model_info.library_name = "transformers"
        mock_hf_api.model_info.return_value = mock_model_info

        # Mock model and tokenizer loading
        mock_model_class.from_pretrained.return_value = mock_classification_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        buffer = ClassificationBuffer()
        buffer.load_model("test/model", timeout=300)

        assert buffer.is_loaded()
        assert buffer.model_name == "test/model"
        assert buffer.is_setfit is False
        assert buffer.model == mock_classification_model
        assert buffer.tokenizer == mock_tokenizer

    @patch("src.text_analysis.text_classification_api.hf_api")
    @patch("src.text_analysis.text_classification_api.SetFitModel")
    def test_load_setfit_model(self, mock_setfit_class, mock_hf_api, mock_setfit_model):
        """Test loading a SetFit model."""
        from src.text_analysis.text_classification_api import ClassificationBuffer

        # Mock HuggingFace API response for SetFit model
        mock_model_info = Mock()
        mock_model_info.library_name = "setfit"
        mock_hf_api.model_info.return_value = mock_model_info

        # Mock SetFit model loading
        mock_setfit_class.from_pretrained.return_value = mock_setfit_model

        buffer = ClassificationBuffer()
        buffer.load_model("test/setfit-model", timeout=300)

        assert buffer.is_loaded()
        assert buffer.model_name == "test/setfit-model"
        assert buffer.is_setfit is True
        assert buffer.model == mock_setfit_model
        assert buffer.tokenizer is None

    @patch("src.text_analysis.text_classification_api.hf_api")
    @patch("src.text_analysis.text_classification_api.AutoModelForSequenceClassification")
    @patch("src.text_analysis.text_classification_api.AutoTokenizer")
    def test_load_same_model_resets_timer(
        self, mock_tokenizer_class, mock_model_class, mock_hf_api
    ):
        """Test that loading the same model again just resets timer."""
        from src.text_analysis.text_classification_api import ClassificationBuffer

        # Mock HuggingFace API
        mock_model_info = Mock()
        mock_model_info.library_name = "transformers"
        mock_hf_api.model_info.return_value = mock_model_info

        buffer = ClassificationBuffer()

        # First load
        buffer.load_model("test/model", timeout=300)
        first_model = buffer.model

        # Second load of same model
        buffer.load_model("test/model", timeout=600)

        # Model should be the same, timeout should be updated
        assert buffer.model is first_model
        assert buffer.timeout == 600

    @patch("src.text_analysis.text_classification_api.hf_api")
    @patch("src.text_analysis.text_classification_api.AutoModelForSequenceClassification")
    @patch("src.text_analysis.text_classification_api.AutoTokenizer")
    @patch("src.text_analysis.text_classification_api.torch")
    def test_predict_proba_standard_model(
        self,
        mock_torch,
        mock_tokenizer_class,
        mock_model_class,
        mock_hf_api,
        mock_classification_model,
        mock_tokenizer,
    ):
        """Test prediction with standard transformer model."""
        from src.text_analysis.text_classification_api import ClassificationBuffer

        # Mock HuggingFace API
        mock_model_info = Mock()
        mock_model_info.library_name = "transformers"
        mock_hf_api.model_info.return_value = mock_model_info

        # Mock model and tokenizer
        mock_model_class.from_pretrained.return_value = mock_classification_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
        mock_classification_model.return_value = mock_output

        # Mock softmax
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.nn.functional.softmax = Mock(
            return_value=torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
        )

        buffer = ClassificationBuffer()
        buffer.load_model("test/model", timeout=300)

        texts = ["This is great!", "This is bad."]
        results = buffer.predict_proba(texts)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    @patch("src.text_analysis.text_classification_api.hf_api")
    @patch("src.text_analysis.text_classification_api.SetFitModel")
    def test_predict_proba_setfit_model(self, mock_setfit_class, mock_hf_api, mock_setfit_model):
        """Test prediction with SetFit model."""
        import numpy as np

        from src.text_analysis.text_classification_api import ClassificationBuffer

        # Mock HuggingFace API
        mock_model_info = Mock()
        mock_model_info.library_name = "setfit"
        mock_hf_api.model_info.return_value = mock_model_info

        # Mock SetFit model
        mock_setfit_class.from_pretrained.return_value = mock_setfit_model
        mock_setfit_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])

        buffer = ClassificationBuffer()
        buffer.load_model("test/setfit-model", timeout=300)

        texts = ["Good text", "Bad text"]
        results = buffer.predict_proba(texts)

        assert isinstance(results, list)
        assert len(results) == 2
        mock_setfit_model.predict_proba.assert_called_once_with(texts)

    def test_predict_proba_without_model_raises_error(self):
        """Test that predict_proba raises error when model not loaded."""
        from src.text_analysis.text_classification_api import ClassificationBuffer

        buffer = ClassificationBuffer()

        with pytest.raises(RuntimeError) as exc_info:
            buffer.predict_proba(["test text"])

        assert "Model not loaded" in str(exc_info.value)


class TestTextClassificationAPIEndpoints:
    """Test text classification API endpoints."""

    @patch("src.text_analysis.text_classification_api.classification_buffer")
    @patch("src.text_analysis.text_classification_api.verify_api_key")
    def test_predict_proba_endpoint(self, mock_verify_key, mock_buffer):
        """Test /predict_proba/ endpoint."""
        from src.text_analysis.text_classification_api import app

        # Mock authentication
        mock_verify_key.return_value = "test-key"

        # Mock buffer
        mock_buffer.load_model = Mock()
        mock_buffer.predict_proba = Mock(
            return_value=[{"negative": 0.1, "neutral": 0.2, "positive": 0.7}]
        )

        client = TestClient(app)
        response = client.post(
            "/predict_proba/",
            json={"text": ["This is great!"], "model": "test/model"},
            headers={"X-API-Key": "test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], list)
        mock_buffer.load_model.assert_called_once_with("test/model")
        mock_buffer.predict_proba.assert_called_once()

    @patch("src.text_analysis.text_classification_api.classification_buffer")
    @patch("src.text_analysis.text_classification_api.verify_api_key")
    def test_predict_proba_endpoint_default_model(self, mock_verify_key, mock_buffer):
        """Test /predict_proba/ endpoint with default model."""
        from src.text_analysis.text_classification_api import app

        # Mock authentication
        mock_verify_key.return_value = "test-key"

        # Mock buffer
        mock_buffer.load_model = Mock()
        mock_buffer.predict_proba = Mock(return_value=[{"negative": 0.3, "positive": 0.7}])

        client = TestClient(app)
        response = client.post(
            "/predict_proba/", json={"text": ["Test text"]}, headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 200
        # Should use default model
        mock_buffer.load_model.assert_called_once()

    @patch("src.text_analysis.text_classification_api.classification_buffer")
    @patch("src.text_analysis.text_classification_api.verify_api_key")
    def test_buffer_status_endpoint(self, mock_verify_key, mock_buffer):
        """Test /buffer_status/ endpoint."""
        from src.text_analysis.text_classification_api import app

        # Mock authentication
        mock_verify_key.return_value = "test-key"

        # Mock buffer status
        mock_buffer.get_status = Mock(
            return_value={
                "is_loaded": True,
                "loaded_at": "2025-01-01T00:00:00",
                "timeout_seconds": 300,
                "timer_active": True,
            }
        )

        client = TestClient(app)
        response = client.get("/buffer_status/", headers={"X-API-Key": "test-key"})

        assert response.status_code == 200
        data = response.json()
        assert data["is_loaded"] is True
        assert "timeout_seconds" in data
        mock_buffer.get_status.assert_called_once()
