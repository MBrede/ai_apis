"""
Buffer class for managing model loading/unloading with automatic timeout.

This module provides an abstract base class for implementing model buffering
with automatic memory cleanup after a specified timeout period. The timer
runs in a separate thread and tracks the last time the model was accessed.

Usage Example:
    class MyModelBuffer(Model_Buffer):
        def load_model(self, model_path: str, timeout: int = 300, **kwargs):
            # Call parent to set up timer
            super().load_model(timeout=timeout)

            # Load your model
            self.model = load_my_model(model_path)
            self.loaded_at = datetime.now()

            # Start the timer
            if self.timer:
                self.timer.start()

        def inference(self, *args, **kwargs):
            # Reset timer on each use
            self.reset_timer()

            # Do inference
            return self.model(*args, **kwargs)
"""

import gc
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock, Timer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class Model_Buffer(ABC):
    """
    Abstract base class for managing ML model loading/unloading with timeout-based memory cleanup.

    This class provides a framework for implementing model buffering with automatic
    resource cleanup after a specified timeout period. It handles CUDA memory management
    and provides hooks for custom model loading logic. The timer runs in a separate thread
    and can be reset on each model access to prevent premature unloading.

    Thread Safety:
        All public methods are thread-safe and can be called from multiple threads.
        The timer runs in a separate daemon thread.

    Attributes:
        timer (Timer | None): Timer for automatic model unloading
        model (Any | None): The loaded ML model
        pipeline (Any | None): The loaded pipeline (if applicable)
        tokenizer (Any | None): The loaded tokenizer (if applicable)
        timeout (int): Timeout duration in seconds (-1 to disable)
        loaded_at (datetime | None): Timestamp when model was loaded
        last_accessed (datetime | None): Timestamp of last model access
    """

    def __init__(self) -> None:
        """Initialize the model buffer with None values for all attributes."""
        self.timer: Timer | None = None
        self.model: Any = None
        self.pipeline: Any = None
        self.tokenizer: Any = None
        self.timeout: int = -1
        self.loaded_at: datetime | None = None
        self.last_accessed: datetime | None = None
        self._lock: Lock | None = None  # Lazy init - don't create until first use (gunicorn fork safety)
        logger.info(f"{self.__class__.__name__} initialized")

    @property
    def lock(self) -> Lock:
        """
        Lazy-initialize lock on first access.

        This is critical for gunicorn compatibility: Lock objects created before
        fork() are broken in child processes. By creating the lock on first access,
        we ensure it's created in the worker process after fork.
        """
        if self._lock is None:
            self._lock = Lock()
        return self._lock

    def is_loaded(self) -> bool:
        """
        Check if a model is currently loaded.

        Returns:
            bool: True if model is loaded, False otherwise
        """
        with self.lock:
            return self.model is not None or self.pipeline is not None

    def get_status(self) -> dict:
        """
        Get current buffer status.

        Returns:
            dict: Status information including loaded state, timestamps, and timeout

        Example:
            >>> buffer.get_status()
            {
                'is_loaded': True,
                'loaded_at': '2025-01-06T10:30:00',
                'last_accessed': '2025-01-06T10:35:00',
                'timeout_seconds': 300,
                'timer_active': True
            }
        """
        with self.lock:
            return {
                "is_loaded": self.is_loaded(),
                "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
                "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
                "timeout_seconds": self.timeout,
                "timer_active": (
                    self.timer is not None and self.timer.is_alive() if self.timer else False
                ),
            }

    def unload_model(self) -> None:
        with self.lock:
            logger.info(f"{self.__class__.__name__}: Unloading model")

            timer_to_cancel = self.timer
            self.timer = None

            self.model = None
            self.pipeline = None
            self.tokenizer = None
            self.loaded_at = None
            self.last_accessed = None

        if timer_to_cancel is not None:
            try:
                timer_to_cancel.cancel()
            except Exception as e:
                logger.warning(f"Error canceling timer: {e}")

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass

    def reset_timer(self, timeout: int | None = None) -> None:
        """
        Reset the unload timer to prevent premature model unloading.

        Call this method every time the model is accessed/used to extend
        its lifetime. The timer is thread-safe and can be reset while active.

        Args:
            timeout: Optional new timeout duration. If None, uses existing timeout.

        Example:
            >>> buffer.reset_timer()  # Reset with current timeout
            >>> buffer.reset_timer(600)  # Reset with new 10-minute timeout
        """
        with self.lock:
            # Update last accessed time
            self.last_accessed = datetime.now()

            # Update timeout if provided
            if timeout is not None:
                self.timeout = timeout

            # Only reset timer if timeout is positive
            if self.timeout <= 0:
                return

            # Cancel existing timer
            if self.timer is not None:
                try:
                    self.timer.cancel()
                except Exception as e:
                    logger.warning(f"Error canceling timer during reset: {e}")

            # Create and start new timer
            self.timer = Timer(self.timeout, self.unload_model)
            self.timer.daemon = True  # Don't prevent program exit
            self.timer.start()
            logger.debug(f"Timer reset: {self.timeout}s until unload")

    def cancel_timer(self) -> None:
        """
        Cancel the unload timer without unloading the model.

        Useful when you want to keep the model loaded indefinitely
        or manage unloading manually.

        Example:
            >>> buffer.cancel_timer()  # Model stays loaded until manual unload
        """
        with self.lock:
            if self.timer is not None:
                try:
                    self.timer.cancel()
                    logger.debug("Timer cancelled")
                except Exception as e:
                    logger.warning(f"Error canceling timer: {e}")
                self.timer = None
                self.timeout = -1

    @abstractmethod
    def load_model(self, *args, timeout: int = 300, **kwargs) -> None:
        """
        Load a model with optional automatic unloading after timeout.

        This abstract method must be implemented by subclasses to define
        custom model loading logic. The implementation should:

        1. Call this parent method to set up the timer
        2. Load the actual model/pipeline/tokenizer
        3. Set self.loaded_at = datetime.now()
        4. Start the timer with self.timer.start() if desired

        Args:
            *args: Variable length argument list for model loading
            timeout (int): Time in seconds before automatic unloading.
                          Use -1 to disable automatic unloading.
                          Use 0 to disable timer (keep forever).
                          Defaults to 300 (5 minutes).
            **kwargs: Arbitrary keyword arguments for model loading

        Example Implementation:
            def load_model(self, model_path: str, timeout: int = 300, device: str = 'cuda'):
                # Set up timer (but don't start yet)
                super().load_model(timeout=timeout)

                # Load model
                if not self.is_loaded():
                    self.model = MyModel.load(model_path).to(device)
                    self.loaded_at = datetime.now()
                    logger.info(f"Model loaded from {model_path}")

                    # Start timer if configured
                    if self.timer:
                        self.timer.start()
                        logger.info(f"Auto-unload timer started: {timeout}s")
                else:
                    logger.info("Model already loaded, resetting timer")
                    self.reset_timer(timeout)
        """
        with self.lock:
            # Cancel any existing timer
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None

            # Set timeout
            self.timeout = timeout

            # Create timer if timeout is positive (but don't start yet - subclass does that)
            if timeout > 0:
                self.timer = Timer(timeout, self.unload_model)
                self.timer.daemon = True
                logger.info(f"Timer configured: {timeout}s until unload")
            else:
                logger.info("Auto-unload disabled (timeout <= 0)")

    def __enter__(self):
        """
        Context manager entry - reset timer on model access.

        Example:
            >>> with buffer:
            ...     result = buffer.model.predict(data)
        """
        if self.is_loaded():
            self.reset_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no action needed."""
        pass

    def __del__(self):
        """Cleanup on deletion - cancel timer and unload model."""
        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.cancel()
        except Exception:
            pass
