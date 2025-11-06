import torch
from abc import ABC, abstractmethod
from threading import Timer

class Model_Buffer(ABC):
    
    def __init__(self):
        self.timer = None
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        
    def unload_model(self):
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        self.timer = None
    
    @abstractmethod
    def load_model(self, *args, timeout: int = 300, **kwargs):
        if timer is not None:
            self.timer.cancel()
        if timeout > -1:
            self.timer = Timer(timeout, self.unload_model)