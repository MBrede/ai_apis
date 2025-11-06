"""
Buffer class for managing model loading/unloading with automatic timeout.

This module provides an abstract base class for implementing model buffering
with automatic memory cleanup after a specified timeout period.
"""

import gc
import torch
from abc import ABC, abstractmethod
from threading import Timer

class Model_Buffer(ABC):
    """
    Abstract base class for managing ML model loading/unloading with timeout-based memory cleanup.

    This class provides a framework for implementing model buffering with automatic
    resource cleanup after a specified timeout period. It handles CUDA memory management
    and provides hooks for custom model loading logic.

    Attributes:
        timer (Timer | None): Timer for automatic model unloading
        model (Any | None): The loaded ML model
        pipeline (Any | None): The loaded pipeline (if applicable)
        tokenizer (Any | None): The loaded tokenizer (if applicable)
    """

    def __init__(self) -> None:
        """Initialize the model buffer with None values for all attributes."""
        self.timer = None
        self.model = None
        self.pipeline = None
        self.tokenizer = None

    def unload_model(self) -> None:
        """
        Unload the model and free GPU memory.

        Clears all model-related attributes, runs garbage collection,
        and empties the CUDA cache to free GPU memory.
        """
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        self.timer = None

    @abstractmethod
    def load_model(self, *args, timeout: int = 300, **kwargs):
        """
        Load a model with optional automatic unloading after timeout.

        This abstract method must be implemented by subclasses to define
        custom model loading logic. The base implementation manages the
        timeout timer for automatic cleanup.

        Args:
            *args: Variable length argument list for model loading
            timeout (int): Time in seconds before automatic unloading.
                          Use -1 to disable automatic unloading. Defaults to 300.
            **kwargs: Arbitrary keyword arguments for model loading
        """
        if self.timer is not None:
            self.timer.cancel()
        if timeout > -1:
            self.timer = Timer(timeout, self.unload_model)