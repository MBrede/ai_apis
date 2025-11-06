# Model Buffer Class - Integration Guide

Complete guide for integrating the `Model_Buffer` class into your API endpoints for automatic GPU memory management.

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Integration Steps](#integration-steps)
- [API Patterns](#api-patterns)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The `Model_Buffer` class provides automatic GPU memory management for ML models by:

- ✅ Loading models on-demand
- ✅ Tracking model usage with timestamps
- ✅ Automatically unloading models after inactivity
- ✅ Thread-safe operations for concurrent API requests
- ✅ Manual control when needed

**Benefits:**
- Frees GPU memory when models aren't being used
- Supports multiple APIs sharing the same GPU
- Prevents out-of-memory errors
- Improves resource utilization

---

## How It Works

```
API Request → Check if loaded → Use model → Reset timer (5min)
                     ↓
              Not loaded → Load model → Start timer (5min)
                                           ↓
                                      Timer expires → Unload model → Free GPU
```

**Thread Model:**
- Main thread: Handles API requests and model inference
- Timer thread: Runs in background, triggers unload after timeout
- Lock protection: Ensures thread-safe access to model

---

## Quick Start

### 1. Create Your Buffer Class

```python
from core.buffer_class import Model_Buffer
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MyModelBuffer(Model_Buffer):
    """Buffer for your specific model."""

    def __init__(self):
        super().__init__()
        self.model_name = None

    def load_model(self, model_name: str, timeout: int = 300, **kwargs):
        """Load model with auto-unload after timeout seconds."""
        # Set up timer (don't start yet)
        super().load_model(timeout=timeout)

        # Load your model
        if not self.is_loaded() or self.model_name != model_name:
            logger.info(f"Loading {model_name}")
            self.model = load_your_model(model_name, **kwargs)
            self.model_name = model_name
            self.loaded_at = datetime.now()

            # Start timer
            if self.timer:
                self.timer.start()
        else:
            # Model already loaded, just reset timer
            self.reset_timer(timeout)

    def inference(self, *args, **kwargs):
        """Run inference and reset timer."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # Reset timer on each use
        self.reset_timer()

        # Your inference code
        return self.model(*args, **kwargs)
```

### 2. Use in Your API

```python
from fastapi import FastAPI, Depends
from auth import verify_api_key

app = FastAPI()

# Create global buffer instance
model_buffer = MyModelBuffer()

@app.post("/predict")
async def predict(data: dict, api_key: str = Depends(verify_api_key)):
    # Ensure model is loaded (loads automatically if needed)
    if not model_buffer.is_loaded():
        model_buffer.load_model("my-model", timeout=300)

    # Run inference (resets timer automatically)
    result = model_buffer.inference(data)

    return {"result": result}

@app.get("/model_status")
async def get_status(api_key: str = Depends(verify_api_key)):
    return model_buffer.get_status()
```

---

## Integration Steps

### Step 1: Import and Inherit

```python
from core.buffer_class import Model_Buffer
from datetime import datetime

class YourBuffer(Model_Buffer):
    def __init__(self):
        super().__init__()
        # Add your custom attributes
        self.your_attribute = None
```

### Step 2: Implement `load_model()`

**Required pattern:**

```python
def load_model(self, model_id: str, timeout: int = 300, **kwargs):
    # 1. Call parent to set up timer
    super().load_model(timeout=timeout)

    # 2. Check if model needs loading
    if not self.is_loaded() or self.your_attribute != model_id:
        # 3. Load your model
        self.model = load_your_model(model_id, **kwargs)
        # Or: self.pipeline = load_pipeline(model_id)
        # Or: self.tokenizer = load_tokenizer(model_id)

        # 4. Set loaded timestamp
        self.loaded_at = datetime.now()

        # 5. Start timer if configured
        if self.timer:
            self.timer.start()
    else:
        # Model already loaded, reset timer
        self.reset_timer(timeout)
```

### Step 3: Call `reset_timer()` on Model Access

**In every method that uses the model:**

```python
def your_inference_method(self, *args, **kwargs):
    if not self.is_loaded():
        raise RuntimeError("Model not loaded")

    # IMPORTANT: Reset timer on each use
    self.reset_timer()

    # Your code here
    result = self.model(*args, **kwargs)
    return result
```

### Step 4: Use in API Endpoints

```python
# Create global instance
buffer = YourBuffer()

@router.post("/endpoint")
async def endpoint(data: Input, api_key: str = Depends(verify_api_key)):
    # Load if needed
    if not buffer.is_loaded():
        buffer.load_model("model-name", timeout=300)

    # Use model (resets timer)
    result = buffer.your_inference_method(data)
    return result
```

---

## API Patterns

### Pattern 1: Always Load On Request (Recommended)

```python
@router.post("/generate")
async def generate(prompt: str, api_key: str = Depends(verify_api_key)):
    # Load model (no-op if already loaded)
    buffer.load_model("my-model", timeout=300)

    # Use model
    result = buffer.generate(prompt)
    return result
```

**Pros:** Simple, handles cold starts automatically
**Cons:** Extra overhead checking if loaded

### Pattern 2: Check Before Loading

```python
@router.post("/generate")
async def generate(prompt: str, api_key: str = Depends(verify_api_key)):
    # Only load if not loaded
    if not buffer.is_loaded():
        buffer.load_model("my-model", timeout=300)

    # Use model
    result = buffer.generate(prompt)
    return result
```

**Pros:** Slightly more efficient
**Cons:** Extra boilerplate

### Pattern 3: Context Manager (For Complex Operations)

```python
@router.post("/generate")
async def generate(prompt: str, api_key: str = Depends(verify_api_key)):
    # Ensure loaded
    if not buffer.is_loaded():
        buffer.load_model("my-model", timeout=300)

    # Context manager auto-resets timer
    with buffer:
        result = buffer.model.generate(prompt)

    return result
```

### Pattern 4: Multiple Models

```python
# Create separate buffers for different models
whisper_buffer = WhisperBuffer()
sd_buffer = StableDiffusionBuffer()

@router.post("/transcribe_and_generate")
async def process(audio: UploadFile, api_key: str = Depends(verify_api_key)):
    # Each buffer manages its own model
    whisper_buffer.load_model("turbo", timeout=300)
    sd_buffer.load_model("sd-2.1", timeout=600)

    # Use both
    text = whisper_buffer.transcribe(audio)
    image = sd_buffer.generate(text)

    return {"text": text, "image": image}
```

---

## Best Practices

### ✅ DO

1. **Always call `reset_timer()` when using the model**
   ```python
   def inference(self, data):
       self.reset_timer()  # ← Important!
       return self.model(data)
   ```

2. **Use appropriate timeout values**
   - Whisper: 300s (5 min) - fast to reload
   - SD: 600s (10 min) - slower to reload
   - Large LLMs: 1800s (30 min) - very slow to reload

3. **Load models on-demand, not at startup**
   ```python
   # Good: Load when needed
   if not buffer.is_loaded():
       buffer.load_model("model")

   # Bad: Load at module import
   # buffer.load_model("model")  # ← Wastes memory
   ```

4. **Add status endpoints for monitoring**
   ```python
   @router.get("/model_status")
   async def status():
       return buffer.get_status()
   ```

5. **Handle model not loaded errors gracefully**
   ```python
   try:
       result = buffer.inference(data)
   except RuntimeError as e:
       logger.error(f"Model error: {e}")
       # Load and retry
       buffer.load_model("model")
       result = buffer.inference(data)
   ```

### ❌ DON'T

1. **Don't forget to start the timer**
   ```python
   def load_model(self, ...):
       super().load_model(timeout=timeout)
       self.model = load_model()
       # Missing: if self.timer: self.timer.start()
   ```

2. **Don't access model without resetting timer**
   ```python
   def inference(self, data):
       # Missing: self.reset_timer()
       return self.model(data)  # Timer not reset!
   ```

3. **Don't use very short timeouts for slow-loading models**
   ```python
   # Bad: Model takes 2min to load, timeout is 1min
   buffer.load_model("big-model", timeout=60)
   ```

4. **Don't load models at module-level**
   ```python
   # Bad: Loads immediately, wastes GPU
   buffer = ModelBuffer()
   buffer.load_model("model")  # ← Don't do this
   ```

5. **Don't ignore thread safety for custom attributes**
   ```python
   class MyBuffer(Model_Buffer):
       def custom_method(self):
           # Bad: Direct access without lock
           self.custom_attr = value

           # Good: Use lock for custom attributes
           with self._lock:
               self.custom_attr = value
   ```

---

## Advanced Features

### Manual Control

```python
# Cancel auto-unload (keep loaded forever)
buffer.cancel_timer()

# Manual unload
buffer.unload_model()

# Change timeout on the fly
buffer.reset_timer(timeout=600)
```

### Status Monitoring

```python
status = buffer.get_status()
# {
#     'is_loaded': True,
#     'loaded_at': '2025-01-06T10:30:00',
#     'last_accessed': '2025-01-06T10:35:00',
#     'timeout_seconds': 300,
#     'timer_active': True
# }
```

### Custom Cleanup

```python
class MyBuffer(Model_Buffer):
    def unload_model(self):
        # Custom cleanup before unload
        if self.model:
            self.model.save_cache()

        # Call parent cleanup
        super().unload_model()

        # Additional cleanup
        self.cleanup_temp_files()
```

---

## Troubleshooting

### Model Unloads Too Quickly

**Problem:** Model unloads before I'm done with multiple requests.

**Solution:**
1. Increase timeout: `buffer.load_model("model", timeout=600)`
2. Ensure `reset_timer()` is called on each request
3. Check timer is starting: Add logging in `load_model()`

### Model Never Unloads

**Problem:** GPU memory stays occupied.

**Solutions:**
1. Check timer is starting: `if self.timer: self.timer.start()`
2. Verify timeout > 0: `buffer.load_model("model", timeout=300)`
3. Check for exceptions preventing unload
4. Manual unload: `buffer.unload_model()`

### Thread Safety Issues

**Problem:** Race conditions or deadlocks.

**Solutions:**
1. Don't hold locks for long operations
2. Use `with self._lock:` only for attribute access
3. Don't call `reset_timer()` inside a lock you already hold
4. The parent class handles locking - don't override unless needed

### Memory Not Freed

**Problem:** GPU memory not released after unload.

**Solutions:**
1. Verify `unload_model()` clears all references:
   ```python
   self.model = None
   self.pipeline = None
   self.tokenizer = None
   ```
2. Check for circular references
3. Force collection: `gc.collect(); torch.cuda.empty_cache()`

---

## Examples

See the `examples/` directory for complete implementations:

- `whisper_buffer_example.py` - Whisper audio transcription
- `stable_diffusion_buffer_example.py` - Stable Diffusion image generation

---

## API Reference

### Methods

#### `is_loaded() -> bool`
Check if model is loaded.

#### `get_status() -> dict`
Get buffer status with timestamps and timer state.

#### `load_model(*args, timeout=300, **kwargs)`
Abstract method - implement in your subclass.

#### `reset_timer(timeout: Optional[int] = None)`
Reset timer to prevent unload. Call on each model use.

#### `cancel_timer()`
Cancel auto-unload. Model stays loaded until manual unload.

#### `unload_model()`
Manually unload model and free GPU memory.

### Context Manager

```python
with buffer:
    result = buffer.model(data)
# Timer reset automatically
```

### Attributes

- `model`: Your ML model
- `pipeline`: Pipeline (if applicable)
- `tokenizer`: Tokenizer (if applicable)
- `timeout`: Current timeout in seconds
- `loaded_at`: When model was loaded
- `last_accessed`: Last time model was used
- `timer`: Active timer (or None)

---

## License

Part of the SD API project.
