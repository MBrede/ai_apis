# Stable Diffusion API

Image generation API with LORA support.

## Endpoints

- `POST /post_config` - Generate images
- `GET /list_loras` - List available LORAs
- `GET /add_new_lora?name=...` - Add LORA from CivitAI (admin only)

## Example

```python
import requests

response = requests.post(
    "http://localhost:1234/post_config",
    headers={"X-API-Key": "your-key"},
    params={
        "prompt": "A beautiful landscape",
        "model_id": "stabilityai/stable-diffusion-2-1",
        "num_inference_steps": 30,
        "count_returned": 1
    }
)
```

## Features

- Automatic GPU memory management (10-minute timeout)
- LORA support with CivitAI integration
- Multiple SD versions supported
- Img2img and inpainting
