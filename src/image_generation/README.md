# Image Generation APIs

High-performance image generation and multi-modal vision APIs powered by Stable Diffusion and UniDiffuser. Features automatic GPU memory management with configurable timeouts.

## Files

### `stable_diffusion_api.py` ✅
**Port:** 1234
**Models:** SD 1.5, SD 2.0/2.1, SDXL 1.0, Flux.1-dev
**GPU:** cuda
**Authentication:** ✅ Required (X-API-Key header)
**Buffer Timeout:** 10 minutes (configurable)

Main Stable Diffusion API with extensive LORA support, multiple model backends, and automatic GPU memory management.

#### Features:
- Text-to-image generation
- Image-to-image transformation
- LORA adapter support (via Civitai)
- Multiple SD model versions
- Custom scheduler support (Euler, etc.)
- Automatic LORA downloading and caching
- Long prompt support (via lpw_stable_diffusion)
- Commercial usage tracking
- 🆕 **Automatic GPU memory management** - Models unload after 10min inactivity
- 🆕 **API key authentication** - Secure access control
- 🆕 **Centralized configuration** - All settings via `core/config.py`

#### Endpoints:
- `POST /post_config` - Generate images (🔐 auth required)
  - Query params: prompt, model_id, torch_dtype, num_inference_steps, etc.
  - Optional file upload for img2img
- `GET /get_available_loras` - List all LORA models with trigger words (🔐 auth)
- `GET /get_available_stable_diffs` - List supported SD models (🔐 auth)
- `POST /add_new_lora?name=<name>` - Add LORA from Civitai (🔐 **admin** auth)
- `GET /buffer_status` - Check model buffer status (🔐 auth)
- `GET /llm_prompt_assistance?text=<text>` - Get LLM help (🔐 auth, forwarded to legacy API)
- `GET /llm_interface?text=<text>` - LLM generation (🔐 auth, forwarded to legacy API)

#### Start Command:
```bash
gunicorn stable_diffusion_api:app --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:1234
```

#### Environment Variables:
- `HF_TOKEN` - HuggingFace API token (in `.env`)
- `CIVIT_KEY` - Civitai API key for LORA downloads (in `.env`)
- `API_KEY` - API key for authentication (in `.env`)
- `ADMIN_API_KEY` - Admin API key for privileged operations (in `.env`)

#### Configuration:
LORA models stored in `lora_list.json`:
```json
{
  "model_name": {
    "model_id": 123456,
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "lora_path": "https://civitai.com/api/download/...",
    "trigger words": ["keyword1", "keyword2"],
    "allow_no_mention": true,
    "usage_rights": "AllowCommercialUse"
  }
}
```

#### Usage Examples:

**Text-to-Image:**
```python
import requests
from PIL import Image
from io import BytesIO

headers = {"X-API-Key": "your-api-key"}
response = requests.post(
    "http://localhost:1234/post_config",
    headers=headers,
    params={
        "prompt": "A serene mountain landscape, oil painting style",
        "model_id": "stabilityai/stable-diffusion-2-1",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 768,
        "height": 768,
        "count_returned": 4,
        "seed": 42
    }
)
image = Image.open(BytesIO(response.content))
image.show()
```

**Image-to-Image:**
```python
files = {'image': ('input.jpg', open('input.jpg', 'rb'), 'multipart/form-data')}
response = requests.post(
    "http://localhost:1234/post_config",
    params={
        "prompt": "Make it sunset",
        "model_id": "stabilityai/stable-diffusion-2-1",
        "num_inference_steps": 20
    },
    files=files
)
```

**Using LORA:**
```python
# First, add the LORA (by name or Civitai ID)
requests.get("http://localhost:1234/add_new_lora?name=pixel_art_style")

# Use it in generation
response = requests.post(
    "http://localhost:1234/post_config",
    params={
        "prompt": "pixel_art_style, a cat wearing sunglasses",
        "lora": "pixel_art_style",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0"
    }
)
```

---

### `unidiffuser_api.py`
**Port:** 8000
**Model:** thu-ml/unidiffuser-v1
**GPU:** cuda

Multi-modal generation supporting text→image, image→text, and image embeddings.

#### Features:
- Text-to-image generation
- Image-to-text (caption generation)
- CLIP and VAE embedding extraction
- Unified architecture for multi-modal tasks

#### Endpoints:
- `POST /prompt2img` - Generate image from text
- `POST /img2prompt` - Generate caption from image
- `POST /img2embed` - Extract embeddings from image

#### Start Command:
```bash
gunicorn unidiffuser_api:app --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

#### Usage Examples:

**Image Captioning:**
```python
files = {'image': ('photo.jpg', open('photo.jpg', 'rb'), 'image/jpeg')}
response = requests.post(
    "http://localhost:8000/img2prompt",
    files=files
)
caption = response.text
print(f"Caption: {caption}")
```

**Extract Embeddings:**
```python
files = {'image': ('photo.jpg', open('photo.jpg', 'rb'), 'image/jpeg')}
response = requests.post(
    "http://localhost:8000/img2embed",
    files=files
)
embeddings = response.json()
print(f"CLIP shape: {len(embeddings['CLIP_embeddings'])}")
print(f"VAE shape: {len(embeddings['vae_emb'])}")
```

---

## Performance Comparison

| Model | Speed | Quality | VRAM | Use Case |
|-------|-------|---------|------|----------|
| SD 1.5 | Fast | Good | 4GB | Quick iterations |
| SD 2.1 | Medium | Better | 6GB | Higher quality |
| SDXL 1.0 | Slower | Best | 12GB | Production quality |
| Flux.1-dev | Slow | Excellent | 24GB | State-of-the-art |
| UniDiffuser | Medium | Good | 8GB | Multi-modal tasks |

## Hardware Requirements

**Minimum (SD 1.5/2.1):**
- GPU: 8GB VRAM (RTX 3070, RTX 4060 Ti)
- RAM: 16GB
- Storage: 20GB

**Recommended (SDXL, Flux):**
- GPU: 24GB+ VRAM (RTX 4090, A5000, A6000)
- RAM: 32GB+
- Storage: 100GB SSD

**Production:**
- GPU: 40GB+ VRAM (A100, H100)
- Multiple GPUs for parallel serving
- NVMe SSD for model storage
- 64GB+ system RAM

## Optimization Tips

1. **Use appropriate precision:**
   - SD 1.5/2.1: float16
   - SDXL: float16 or bfloat16
   - Flux: bfloat16

2. **Enable optimizations:**
   ```python
   pipe.enable_attention_slicing()
   pipe.enable_vae_slicing()
   pipe.enable_vae_tiling()  # For high-res
   ```

3. **Use compiled models:**
   ```python
   pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
   ```

4. **Batch processing:**
   Generate multiple images in one call using `count_returned`

5. **LORA over full models:**
   Faster loading and less VRAM than switching full models

## Known Issues

1. ~~**Hardcoded IPs**~~ ✅ Fixed - Now in `core/config.py`
2. **No request queuing** - concurrent requests may cause issues
3. **LORA downloads** can timeout on slow connections
4. ~~**idefics_api.py incomplete**~~ ✅ Removed
5. **No image validation** - malformed images can crash the API
6. ~~**Memory leaks with LORA**~~ ✅ Fixed - Buffer class cleans up properly

## Troubleshooting

**Out of Memory:**
```python
# Reduce batch size
count_returned = 1

# Lower resolution
width, height = 512, 512

# Enable memory efficient attention
pipe.enable_attention_slicing("max")
```

**Slow Generation:**
- Reduce `num_inference_steps` (try 20-30)
- Use lighter models (SD 1.5 instead of SDXL)
- Enable torch.compile()

**LORA Not Working:**
- Check base_model matches in lora_list.json
- Verify LORA files downloaded to `loras/` directory
- Check Civitai API key is valid

## Future Improvements

- [ ] Add request queue for concurrent handling
- [ ] Implement model warm-up on startup
- [ ] Add image validation and preprocessing
- [ ] Support ControlNet and IP-Adapter
- [ ] Add negative embeddings support
- [ ] Implement multi-GPU support
- [ ] Add EXIF metadata to generated images
- [ ] Better error messages for common issues
- [ ] Add image upscaling endpoint
- [ ] Support for video generation (AnimateDiff)
- [ ] Add safety checker option (NSFW filter)
- [ ] Implement A/B testing for prompts

## License Considerations

- **Stable Diffusion models**: Check individual model licenses
- **LORA models**: Respect Civitai creator licenses
- **Commercial use**: Verify `usage_rights` in lora_list.json
- **Attribution**: Some models require credit (`allow_no_mention`)

Always review model cards on HuggingFace and Civitai before commercial deployment.
