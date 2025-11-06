# LLM APIs

⚠️ **Note: These APIs use older LLM models that may be outdated.**

## Overview

This folder contains FastAPI services for running Large Language Models, specifically focused on generating detailed image descriptions for Stable Diffusion models.

## ⚠️ Outdated Models Warning

The current implementations use:
- **Mixtral-8x7B-Instruct-v0.1** (2023 model)
- **c4ai-command-r-v01-4bit** (older Cohere model)

### Recommended Upgrades

Consider upgrading to newer, more capable models:

**Open Source:**
- Llama 3.3 70B/405B (Meta, 2024)
- Qwen 2.5 72B/110B (Alibaba, 2024)
- Mixtral-8x22B-Instruct-v0.1 (2024)
- DeepSeek-V3 (2025)

**Closed Source (via API):**
- GPT-4 Turbo / GPT-4o
- Claude 3.5 Sonnet
- Gemini 2.0 Flash

## Files

### `huggingface_api.py`
**Port:** 8000
**Model:** Mixtral-8x7B-Instruct-v0.1 (4-bit quantized)
**GPU:** cuda:1

#### Endpoints:
- `GET /llm_prompt_assistance?text=<text>` - Generate detailed image descriptions
- `GET /llm_interface?text=<text>&role=<role>&temp=<temp>` - General text generation
- `POST /answer/` - Structured queries with system prompts

#### Start Command:
```bash
gunicorn huggingface_api:app -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 -t 30000
```

#### Environment Variables:
- `hf_token` - HuggingFace API token for model access

### `command_r_api.py`
**Port:** 1234
**Model:** CohereForAI/c4ai-command-r-v01-4bit
**GPU:** cuda

#### Endpoints:
- `GET /llm_prompt_assistance?text=<text>` - Generate image descriptions
- `GET /llm_interface?text=<text>&role=<role>&temp=<temp>` - General text generation
- `POST /answer/` - Structured queries

#### Start Command:
```bash
gunicorn command_r_api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:1234
```

### `llm_wrapper.py`
**Port:** 8080
**Purpose:** Unified endpoint manager for multiple LLM instances

#### Features:
- Dynamic LLM registration
- Automatic health checking
- Load balancing across multiple LLM instances
- JSON output support (non-Unsloth models only)

#### Endpoints:
- `GET /list_available_llms` - List healthy LLM endpoints
- `POST /register_llm` - Register a new LLM endpoint
- `POST /llm_answer/` - Query any registered LLM
- `POST /json_answer/` - Get structured JSON output

#### Start Command:
```bash
gunicorn llm_wrapper:app -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 -t 30000
```

#### Configuration:
Edit `available_endpoints.json` to add/remove LLM endpoints:
```json
{
  "LLM": {
    "mixtral": {
      "name": "mixtral",
      "source": "huggingface",
      "IP": "149.222.209.66",
      "Port": 8000,
      "Unsloth": false,
      "Worker_Count": 1
    }
  }
}
```

## Usage Examples

### Generate Image Description
```python
import requests

response = requests.get(
    "http://localhost:8000/llm_prompt_assistance",
    params={"text": "A beautiful sunset over mountains"}
)
print(response.text)
# Output: "Description: A (majestic mountain range:1.3) at dusk..."
```

### General Text Generation
```python
import requests

response = requests.get(
    "http://localhost:8000/llm_interface",
    params={
        "text": "Explain quantum computing",
        "role": "You are a helpful physics teacher",
        "temp": 0.7
    }
)
print(response.text)
```

### Structured Query
```python
import requests

response = requests.post(
    "http://localhost:8000/answer/",
    json={
        "system": "You are a creative writer",
        "messages": "Write a short poem about AI",
        "temperature": 0.8,
        "max_tokens": 200
    }
)
print(response.json()["message"])
```

## Performance Considerations

### Current Setup:
- **4-bit quantization** reduces memory but limits precision
- **Single GPU** per model
- **No batching** implemented

### Optimization Opportunities:
1. **Upgrade to newer models** with better efficiency
2. **Implement Flash Attention 2** for 2-4x speedup
3. **Use vLLM** for production serving (much faster)
4. **Add request batching** for throughput
5. **Implement caching** for repeated prompts

## Migration Guide

To upgrade models:

1. **Update model name** in the `__init__` method
2. **Adjust quantization** if needed (newer models may need 8-bit)
3. **Update prompt template** for model-specific formatting
4. **Test generation quality** and adjust parameters

Example for Llama 3.3:
```python
model_name = 'meta-llama/Llama-3.3-70B-Instruct'

# May need to adjust the prompt format
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

## Known Issues

1. ✅ **Fixed**: `huggingface_api.py:112` - typo `item.message` changed to `item.messages`
2. **Hardcoded IPs** - Should use environment variables
3. **No authentication** - APIs are open to anyone
4. **Limited error handling** - Needs better exception management
5. **No request logging** - Should add logging for monitoring

## Hardware Requirements

**Minimum:**
- GPU: 16GB VRAM (for 4-bit models)
- RAM: 32GB system RAM
- Storage: 50GB for model weights

**Recommended:**
- GPU: 40GB+ VRAM (A100, H100)
- RAM: 64GB+ system RAM
- Storage: 200GB SSD
- Multiple GPUs for serving multiple models

## Dependencies

See `pyproject.toml` for complete list. Key dependencies:
- transformers >= 4.57.1
- torch >= 2.6.0
- fastapi >= 0.115.8
- bitsandbytes >= 0.45.0 (for quantization)

## Future Improvements

- [ ] Upgrade to newer LLM models
- [ ] Implement vLLM for production serving
- [ ] Add authentication and rate limiting
- [ ] Implement request batching
- [ ] Add response caching
- [ ] Better error handling and logging
- [ ] Add model switching without restart
- [ ] Implement streaming responses
- [ ] Add token counting and billing
- [ ] Multi-GPU support
