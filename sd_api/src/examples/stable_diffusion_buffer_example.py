"""
Example: Using Model_Buffer with Stable Diffusion API

This shows how to integrate the buffer class into stable_diffusion_api.py for
automatic model unloading after inactivity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from diffusers import StableDiffusionPipeline
from core.buffer_class import Model_Buffer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StableDiffusionBuffer(Model_Buffer):
    """
    Stable Diffusion pipeline buffer with automatic unloading.

    Keeps the SD pipeline in GPU memory and automatically unloads
    it after a specified timeout of inactivity.
    """

    def __init__(self):
        super().__init__()
        self.model_id: str | None = None
        self.torch_dtype = torch.float16

    def load_model(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        timeout: int = 600,  # 10 minutes default for SD
        torch_dtype = torch.float16,
        device: str = "cuda",
        **kwargs
    ) -> None:
        """
        Load Stable Diffusion pipeline with automatic unloading.

        Args:
            model_id: HuggingFace model ID
            timeout: Seconds until auto-unload (default: 600)
            torch_dtype: Data type for model weights
            device: Device to load model on
            **kwargs: Additional pipeline arguments
        """
        # Set up timer
        super().load_model(timeout=timeout)

        # Load pipeline if not already loaded or if different model requested
        if not self.is_loaded() or self.model_id != model_id:
            logger.info(f"Loading Stable Diffusion: {model_id}")

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                **kwargs
            )
            self.pipeline = self.pipeline.to(device)

            self.model_id = model_id
            self.torch_dtype = torch_dtype
            self.loaded_at = datetime.now()
            logger.info(f"SD pipeline loaded successfully: {model_id}")

            # Start timer
            if self.timer:
                self.timer.start()
                logger.info(f"Auto-unload timer started: {timeout}s")
        else:
            # Pipeline already loaded, just reset timer
            logger.info(f"SD pipeline already loaded: {model_id}, resetting timer")
            self.reset_timer(timeout)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ):
        """
        Generate image and reset timer.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            **kwargs: Additional pipeline arguments

        Returns:
            Generated image(s)
        """
        if not self.is_loaded():
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        # Reset timer on each use
        self.reset_timer()

        logger.info(f"Generating image: {prompt[:50]}...")

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

        logger.info("Image generation complete")
        return result.images


# ============================================================================
# Integration Example for stable_diffusion_api.py
# ============================================================================

"""
To integrate into stable_diffusion_api.py:

1. Replace DiffusionModel class with buffer-based version:

   class DiffusionModel(Model_Buffer):
       def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
           super().__init__()
           self.loaded_lora = None
           self.config = {'torch_dtype': torch.float16}
           self.type = 'prompt2img'
           self.model_id = None
           self.load_model({'model_id': model_id, 'type': 'prompt2img'}, timeout=600)

       def load_model(self, config: dict, timeout: int = 600):
           # Set up timer
           super().load_model(timeout=timeout)

           # Your existing load logic here...

           # Start timer after loading
           if self.timer:
               self.timer.start()

       def gen_image(self, prompt, config):
           # Reset timer on each generation
           self.reset_timer()

           # Your existing generation logic...

2. Optional: Add buffer status endpoint:

   @router.get("/model_status")
   async def get_model_status(api_key: str = Depends(verify_api_key)):
       return {
           "buffer_status": model.get_status(),
           "current_model": model.model_id,
           "loaded_lora": model.loaded_lora
       }

3. Optional: Add timeout configuration endpoint:

   @router.post("/set_timeout")
   async def set_timeout(timeout: int, api_key: str = Depends(verify_admin_key)):
       model.reset_timer(timeout)
       return {"message": f"Timeout set to {timeout}s"}

4. Optional: Manual unload endpoint:

   @router.post("/unload_model")
   async def unload_model_endpoint(api_key: str = Depends(verify_admin_key)):
       model.unload_model()
       return {"message": "Model unloaded"}
"""


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Create buffer
    buffer = StableDiffusionBuffer()

    # Load model with 10-minute timeout
    buffer.load_model("stabilityai/stable-diffusion-2-1", timeout=600)

    # Check status
    print("Status:", buffer.get_status())

    # Generate image (resets timer automatically)
    images = buffer.generate_image(
        prompt="a beautiful sunset over mountains",
        num_inference_steps=30
    )
    print(f"Generated {len(images)} image(s)")

    # Generate another (resets timer again)
    # images = buffer.generate_image("a cat wearing sunglasses")

    # Use as context manager
    with buffer:
        # images = buffer.pipeline("robot in space")
        pass

    # Cancel auto-unload to keep model loaded
    # buffer.cancel_timer()

    # Manual unload
    # buffer.unload_model()
