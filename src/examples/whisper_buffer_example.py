"""
Example: Using Model_Buffer with Whisper API

This shows how to integrate the buffer class into whisper_api.py for
automatic model unloading after inactivity.
"""

import os
import whisper
from src.core.buffer_class import Model_Buffer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperBuffer(Model_Buffer):
    """
    Whisper model buffer with automatic unloading.

    Keeps the Whisper model in GPU memory and automatically unloads
    it after a specified timeout of inactivity.
    """

    def __init__(self):
        super().__init__()
        self.model_name: str = "base"

    def load_model(self, model_name: str = "turbo", timeout: int = 300, **kwargs) -> None:
        """
        Load Whisper model with automatic unloading.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, turbo)
            timeout: Seconds until auto-unload (default: 300)
            **kwargs: Additional whisper.load_model arguments
        """
        # Set up timer
        super().load_model(timeout=timeout)

        # Load model if not already loaded or if different model requested
        if not self.is_loaded() or self.model_name != model_name:
            logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name, **kwargs)
            self.model_name = model_name
            self.loaded_at = datetime.now()
            logger.info(f"Whisper {model_name} loaded successfully")

            # Start timer
            if self.timer:
                self.timer.start()
                logger.info(f"Auto-unload timer started: {timeout}s")
        else:
            # Model already loaded, just reset timer
            logger.info(f"Whisper {model_name} already loaded, resetting timer")
            self.reset_timer(timeout)

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """
        Transcribe audio file and reset timer.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional whisper.transcribe arguments

        Returns:
            dict: Transcription result
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reset timer on each use
        self.reset_timer()

        logger.info(f"Transcribing: {audio_path}")
        result = self.model.transcribe(audio_path, **kwargs)
        logger.info(f"Transcription complete")

        return result


# ============================================================================
# Integration Example for whisper_api.py
# ============================================================================

"""
To integrate into whisper_api.py:

1. Create global buffer instance:
   whisper_buffer = WhisperBuffer()

2. Modify /transcribe/ endpoint:

   @router.post("/transcribe/")
   async def transcribe(file: UploadFile, model_to_use: str = 'turbo',
                       api_key: str = Depends(verify_api_key)):
       # Ensure model is loaded
       if not whisper_buffer.is_loaded() or whisper_buffer.model_name != model_to_use:
           whisper_buffer.load_model(model_to_use, timeout=300)

       # Save uploaded file
       with open(file.filename, 'wb') as f:
           file_contents = await file.read()
           f.write(file_contents)

       # Transcribe (automatically resets timer)
       answer = whisper_buffer.transcribe(file.filename, verbose=False)
       os.remove(file.filename)

       return {"answer": answer['text']}

3. Optional: Add status endpoint:

   @router.get("/model_status")
   async def get_model_status(api_key: str = Depends(verify_api_key)):
       return whisper_buffer.get_status()

4. Optional: Add manual unload endpoint (admin only):

   @router.post("/unload_model")
   async def unload_model(api_key: str = Depends(verify_admin_key)):
       whisper_buffer.unload_model()
       return {"message": "Model unloaded"}
"""


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Create buffer
    buffer = WhisperBuffer()

    # Load model with 5-minute timeout
    buffer.load_model("turbo", timeout=300)

    # Check status
    print("Status:", buffer.get_status())

    # Transcribe audio (resets timer automatically)
    # result = buffer.transcribe("audio.mp3")
    # print(result['text'])

    # Use as context manager (auto-resets timer)
    with buffer:
        # result = buffer.model.transcribe("audio.mp3")
        pass

    # Cancel timer to keep model loaded
    # buffer.cancel_timer()

    # Manual unload
    # buffer.unload_model()
