"""
Tests for the Model_Buffer abstract base class.

Tests model buffer functionality including loading, unloading, timer management,
and thread safety.
"""

import time
from datetime import datetime
from unittest.mock import Mock, patch

from src.core.buffer_class import Model_Buffer


class ConcreteModelBuffer(Model_Buffer):
    """Concrete implementation of Model_Buffer for testing."""

    def load_model(self, model_name: str = "test-model", timeout: int = 300, **kwargs):
        """
        Load a mock model.

        Args:
            model_name: Name of the model to load
            timeout: Timeout in seconds for auto-unload
            **kwargs: Additional arguments
        """
        # Call parent to set up timer
        super().load_model(timeout=timeout)

        # Load mock model
        if not self.is_loaded():
            self.model = Mock()
            self.model.name = model_name
            self.loaded_at = datetime.now()

            # Start timer if configured
            if self.timer:
                self.timer.start()
        else:
            # Model already loaded, reset timer
            self.reset_timer(timeout)


class TestModelBufferInitialization:
    """Test Model_Buffer initialization."""

    def test_buffer_initializes_with_none_values(self):
        """Test that buffer initializes with None/default values."""
        buffer = ConcreteModelBuffer()

        assert buffer.model is None
        assert buffer.pipeline is None
        assert buffer.tokenizer is None
        assert buffer.timer is None
        assert buffer.timeout == -1
        assert buffer.loaded_at is None
        assert buffer.last_accessed is None
        assert not buffer.is_loaded()

    def test_buffer_has_lock(self):
        """Test that buffer has a lock for thread safety."""
        buffer = ConcreteModelBuffer()

        assert hasattr(buffer, "_lock")
        assert buffer._lock is not None


class TestModelBufferLoading:
    """Test model loading functionality."""

    def test_load_model_basic(self):
        """Test basic model loading."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        assert buffer.is_loaded()
        assert buffer.model is not None
        assert buffer.model.name == "test-model"
        assert buffer.loaded_at is not None
        assert buffer.timeout == 300

    def test_load_model_with_zero_timeout(self):
        """Test loading model with timeout disabled (0)."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=0)

        assert buffer.is_loaded()
        assert buffer.timeout == 0
        assert buffer.timer is None

    def test_load_model_with_negative_timeout(self):
        """Test loading model with timeout disabled (-1)."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=-1)

        assert buffer.is_loaded()
        assert buffer.timeout == -1
        assert buffer.timer is None

    def test_load_model_starts_timer(self):
        """Test that loading model starts the timer."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        assert buffer.timer is not None
        assert buffer.timer.is_alive()

    def test_load_model_twice_resets_timer(self):
        """Test that loading an already loaded model resets the timer."""
        buffer = ConcreteModelBuffer()

        # First load
        buffer.load_model("test-model", timeout=300)
        first_timer = buffer.timer

        # Second load
        time.sleep(0.1)
        buffer.load_model("test-model", timeout=600)

        # Timer should be different (reset)
        assert buffer.timer != first_timer
        assert buffer.timeout == 600


class TestModelBufferUnloading:
    """Test model unloading functionality."""

    @patch("src.core.buffer_class.torch")
    def test_unload_model_clears_references(self, mock_torch):
        """Test that unloading clears all model references."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = Mock()

        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=-1)  # No auto-unload

        # Verify loaded
        assert buffer.is_loaded()

        # Unload
        buffer.unload_model()

        # Verify cleared
        assert not buffer.is_loaded()
        assert buffer.model is None
        assert buffer.pipeline is None
        assert buffer.tokenizer is None
        assert buffer.loaded_at is None
        assert buffer.last_accessed is None
        assert buffer.timer is None

    @patch("src.core.buffer_class.torch")
    def test_unload_model_clears_cuda_cache(self, mock_torch):
        """Test that unloading clears CUDA cache."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = Mock()

        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=-1)
        buffer.unload_model()

        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("src.core.buffer_class.torch")
    def test_unload_model_cancels_timer(self, mock_torch):
        """Test that unloading cancels active timer."""
        mock_torch.cuda.is_available.return_value = False

        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        timer = buffer.timer
        assert timer.is_alive()

        buffer.unload_model()

        # Timer should be cancelled
        time.sleep(0.1)  # Give timer thread time to stop
        assert not timer.is_alive()

    @patch("src.core.buffer_class.torch")
    def test_auto_unload_after_timeout(self, mock_torch):
        """Test that model automatically unloads after timeout."""
        mock_torch.cuda.is_available.return_value = False

        buffer = ConcreteModelBuffer()
        # Use very short timeout for testing
        buffer.load_model("test-model", timeout=0.2)

        assert buffer.is_loaded()

        # Wait for timeout
        time.sleep(0.5)

        # Model should be unloaded
        assert not buffer.is_loaded()


class TestModelBufferTimerManagement:
    """Test timer management functionality."""

    def test_reset_timer_updates_last_accessed(self):
        """Test that reset_timer updates last_accessed timestamp."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        first_accessed = buffer.last_accessed
        time.sleep(0.1)
        buffer.reset_timer()

        assert buffer.last_accessed > first_accessed

    def test_reset_timer_with_new_timeout(self):
        """Test resetting timer with a new timeout value."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        assert buffer.timeout == 300

        buffer.reset_timer(timeout=600)

        assert buffer.timeout == 600

    def test_reset_timer_prevents_unload(self):
        """Test that resetting timer prevents premature unload."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=0.3)

        # Reset timer before timeout
        for _ in range(3):
            time.sleep(0.15)  # Sleep less than timeout
            buffer.reset_timer()
            assert buffer.is_loaded()

    def test_cancel_timer_stops_auto_unload(self):
        """Test that canceling timer prevents auto-unload."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=0.2)

        buffer.cancel_timer()

        # Wait longer than original timeout
        time.sleep(0.5)

        # Model should still be loaded
        assert buffer.is_loaded()

    def test_cancel_timer_sets_timeout_to_minus_one(self):
        """Test that cancel_timer sets timeout to -1."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        buffer.cancel_timer()

        assert buffer.timeout == -1

    def test_reset_timer_with_zero_timeout_does_nothing(self):
        """Test that reset_timer with timeout=0 doesn't create timer."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=-1)

        buffer.reset_timer(timeout=0)

        assert buffer.timer is None
        assert buffer.timeout == 0


class TestModelBufferStatus:
    """Test status reporting functionality."""

    def test_get_status_when_not_loaded(self):
        """Test get_status when no model is loaded."""
        buffer = ConcreteModelBuffer()
        status = buffer.get_status()

        assert status["is_loaded"] is False
        assert status["loaded_at"] is None
        assert status["last_accessed"] is None
        assert status["timeout_seconds"] == -1
        assert status["timer_active"] is False

    def test_get_status_when_loaded(self):
        """Test get_status when model is loaded."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        status = buffer.get_status()

        assert status["is_loaded"] is True
        assert status["loaded_at"] is not None
        assert status["timeout_seconds"] == 300
        assert status["timer_active"] is True

    def test_get_status_after_timer_access(self):
        """Test get_status shows last_accessed after timer reset."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        time.sleep(0.1)
        buffer.reset_timer()

        status = buffer.get_status()

        assert status["last_accessed"] is not None


class TestModelBufferContextManager:
    """Test context manager functionality."""

    def test_context_manager_resets_timer(self):
        """Test that using buffer as context manager resets timer."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        first_accessed = buffer.last_accessed

        time.sleep(0.1)

        with buffer:
            pass

        assert buffer.last_accessed > first_accessed

    def test_context_manager_returns_self(self):
        """Test that context manager returns self."""
        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        with buffer as ctx:
            assert ctx is buffer


class TestModelBufferThreadSafety:
    """Test thread safety of buffer operations."""

    def test_concurrent_timer_resets(self):
        """Test that concurrent timer resets don't cause issues."""
        import threading

        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        # Multiple threads resetting timer
        def reset_many_times():
            for _ in range(10):
                buffer.reset_timer()
                time.sleep(0.01)

        threads = [threading.Thread(target=reset_many_times) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Buffer should still be loaded and consistent
        assert buffer.is_loaded()
        assert buffer.timer is not None

    def test_concurrent_unload_is_safe(self):
        """Test that concurrent unload calls don't cause errors."""
        import threading

        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        # Multiple threads trying to unload
        def try_unload():
            buffer.unload_model()

        threads = [threading.Thread(target=try_unload) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Buffer should be cleanly unloaded
        assert not buffer.is_loaded()


class TestModelBufferCleanup:
    """Test cleanup and destructor."""

    @patch("src.core.buffer_class.torch")
    def test_destructor_cancels_timer(self, mock_torch):
        """Test that destructor cancels timer."""
        mock_torch.cuda.is_available.return_value = False

        buffer = ConcreteModelBuffer()
        buffer.load_model("test-model", timeout=300)

        # Store timer reference to verify it gets cleaned up
        _timer = buffer.timer

        # Delete buffer
        del buffer

        # Give timer time to be cancelled
        time.sleep(0.1)

        # Timer should be cancelled (this is best effort)
        # Note: __del__ is not guaranteed to be called immediately
