# AI APIs Collection

A comprehensive collection of FastAPI-based microservices for running various AI models including LLMs, Stable Diffusion, audio transcription, and text analysis. Features automatic GPU memory management with configurable model unloading.

## 📁 Project Structure

```
ai_apis/
├── src/
│   ├── core/               # Core utilities and shared infrastructure
│   │   ├── auth.py         # 🔐 API key authentication
│   │   ├── config.py       # ⚙️ Centralized configuration
│   │   ├── buffer_class.py # 🧠 Auto GPU memory management
│   │   ├── bot.py          # 🤖 Telegram bot (uses OLLAMA)
│   │   └── api_request.py  # HTTP client utilities
│   ├── image_generation/   # 🎨 Stable Diffusion, UniDiffuser
│   ├── audio/              # 🎙️ Whisper transcription + diarization
│   ├── text_analysis/      # 📝 Sentiment & classification
│   ├── llm/                # 💬 Legacy LLM APIs (backward compat)
│   ├── training/           # 🏋️ Model fine-tuning (Unsloth)
│   └── examples/           # Buffer integration examples
├── pyproject.toml          # Python 3.13+ dependencies
├── .env.example            # Configuration template
├── README.md               # This file
└── automated_cuda_install.yml
```

## ⚠️ Important Notes

### OLLAMA Integration 🆕
The Telegram bot now uses **OLLAMA** for all LLM operations:
- Endpoint: `149.222.209.66:2345`
- Default model: `llama3.3`
- Configurable via `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL` in `.env`

### LLM APIs - Legacy Status
The LLM APIs in `src/llm/` use **older model versions** and are kept for **backward compatibility only**:
- Mixtral-8x7B-Instruct-v0.1 (2023)
- c4ai-command-r-v01-4bit (old version)

**Modern LLM access:** Use OLLAMA endpoint for current models.

### Recent Changes ✅
1. ✅ **Project restructured** - Removed confusing `sd_api/` folder, cleaner hierarchy
2. ✅ **GPU memory management** - All APIs use Model_Buffer for automatic unloading
3. ✅ **Authentication added** - All APIs now require API key
4. ✅ **Centralized configuration** - All IPs/ports in `core/config.py`
5. ✅ **Bot uses OLLAMA** - Switched from legacy LLM APIs
6. ✅ **Fixed bugs** & removed outdated code:
   - Deleted duplicate `whisper/` folder
   - Fixed `buffer_class.py`: Added missing `import gc`
   - Fixed `huggingface_api.py`: Typo `item.message` → `item.messages`
   - Removed incomplete `idefics_api`

## 🚀 Installation

### Prerequisites
- Python 3.13+
- CUDA 12.1+ (for GPU support)
- FFmpeg (for audio processing)

### Setup

```bash
# Clone the repository
cd ai_apis

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For full installation with CUDA
pip install -e ".[all]"

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration:
# - API_KEY (REQUIRED for authentication!)
# - ADMIN_API_KEY (for admin operations)
# - HF_TOKEN (HuggingFace)
# - CIVIT_KEY (Civitai for LORA models)
# - TELEGRAM_TOKEN (Telegram Bot API)
# - OLLAMA_HOST/PORT (LLM endpoint)
# - SD_HOST/PORT (Stable Diffusion endpoint)
# - WHISPER_HOST/PORT (Audio transcription endpoint)
```

## 🔐 Authentication

All APIs are protected by API key authentication:

```bash
# Make requests with X-API-Key header
curl -H "X-API-Key: your-api-key" http://localhost:1234/get_available_stable_diffs

# Python example
headers = {"X-API-Key": "your-api-key"}
response = requests.get("http://localhost:1234/get_available_stable_diffs", headers=headers)
```

**Security Notes:**
- Set `API_KEY` in `.env` before running APIs
- Use `ADMIN_API_KEY` for privileged operations (adding LORA models)
- Set `REQUIRE_AUTH=False` only for local development
- Always use HTTPS in production

## 📚 API Modules

### Core (`src/core/`)
- **config.py** - 🆕 Centralized configuration management
- **auth.py** - 🆕 API key authentication
- **buffer_class.py** - 🆕 Abstract base class for automatic GPU memory management
- **bot.py** - Telegram bot (now uses OLLAMA for LLM)
- **api_request.py** - Client utilities for making API requests

**GPU Memory Management:**
All ML APIs now use the `Model_Buffer` class for automatic model unloading:
- ⏱️ **Configurable timeouts** per model type (SD: 10min, Whisper: 5min, Text: 5min)
- 🔄 **Automatic timer reset** on each model access
- 🧵 **Thread-safe** operations for concurrent requests
- 🗑️ **Smart cleanup** with garbage collection + CUDA cache clearing
- 📊 **Monitoring** via `/buffer_status` endpoints
- See `src/core/BUFFER_CLASS_GUIDE.md` for integration details

### LLM (`src/llm/`) ⚠️ Legacy - For Backward Compatibility Only
- **huggingface_api.py** - Mixtral-8x7B API (old model, port 8000)
- **command_r_api.py** - Cohere Command-R API (old model, port 1234)
- **llm_wrapper.py** - Unified LLM endpoint manager

**Note:** Telegram bot now uses **OLLAMA** (149.222.209.66:2345) for all LLM operations. The legacy LLM APIs are kept for backward compatibility only.

### Image Generation (`src/image_generation/`)
- **stable_diffusion_api.py** - Main SD API with LORA support (port 1234) ✅ Auth enabled
- **unidiffuser_api.py** - Multi-modal generation (text↔image) (port 8000)

### Audio (`src/audio/`)
- **whisper_api.py** - Speech transcription with speaker diarization (port 8080) ✅ Auth enabled

### Text Analysis (`src/text_analysis/`)
- **sentiment_api.py** - Sentiment analysis
- **text_classification_api.py** - General text classification with SetFit

### Training (`src/training/`)
- **retrain_unsloth.py** - Fine-tuning script using Unsloth

## 🔧 Running the APIs

Each API can be started independently:

```bash
# Stable Diffusion API
cd src/image_generation
gunicorn stable_diffusion_api:app --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:1234

# LLM API (Mixtral)
cd src/llm
gunicorn huggingface_api:app -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 -t 30000

# Whisper API
cd src/audio
gunicorn whisper_api:app -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 -t 30000

# Telegram Bot
cd src/core
python bot.py
```

## 📝 API Documentation

Once an API is running, visit `http://localhost:<port>/docs` for interactive Swagger documentation.

### Example Endpoints

**Stable Diffusion:**
- `POST /post_config` - Generate images from text/images
- `GET /get_available_loras` - List available LORA models
- `GET /get_available_stable_diffs` - List SD models
- `GET /buffer_status` - Check model buffer status

**LLM:**
- `GET /llm_interface?text=...` - Generate text
- `GET /llm_prompt_assistance?text=...` - Generate image prompts
- `POST /answer/` - Structured LLM queries

**Whisper:**
- `POST /transcribe/` - Transcribe audio
- `POST /transcribe_and_diarize/` - Transcribe with speaker identification
- `GET /buffer_status` - Check Whisper & diarization buffer status

**Text Analysis:**
- `POST /predict_sentiment/` - Analyze sentiment
- `POST /predict_proba/` - Text classification probabilities
- `GET /buffer_status` - Check model buffer status

## 🤖 Telegram Bot

The bot provides:
- Image generation from text prompts
- Image-to-image transformation
- LLM chat interface
- Audio transcription
- Image description generation

Commands:
- `/start` - Initialize bot
- `/help` - Show all commands
- `/llm` - Switch between SD and LLM mode
- `/set_parameters` - Configure generation settings
- `/get_loras` - List available LORA models

## 🔐 Security Recommendations

1. **Move hardcoded IPs to environment variables**
2. **Add authentication** to all APIs
3. **Implement rate limiting**
4. **Use HTTPS** in production
5. **Secure token storage** (use secrets manager)
6. **Validate user inputs**

## 🛠️ Development

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
ruff check src/

# Run tests (if available)
pytest
```

## 📦 Dependencies

See `pyproject.toml` for complete dependency list. All packages are Python 3.13 compatible.

**Key dependencies:**
- FastAPI 0.115.8+ (API framework)
- PyTorch 2.6.0+ (ML backend)
- Transformers 4.57.1+ (HuggingFace models)
- Diffusers 0.31.0+ (Stable Diffusion)
- OpenAI Whisper 20250625+ (Audio transcription)
- python-telegram-bot 22.5+ (Telegram integration)

## ⚡ Performance Tips

1. **Use 4-bit quantization** for large models (already implemented)
2. **Enable Flash Attention 2** for faster inference
3. **Use LORA adapters** instead of full fine-tuning
4. **Cache models** between requests
5. **Use multiple workers** for non-GPU endpoints

## 📄 License

[Specify your license here]

## 🤝 Contributing

See individual module READMEs for specific contribution guidelines.

## 📞 Support

For issues and questions:
1. Check module-specific READMEs
2. Review API documentation at `/docs`
3. Submit issues on GitHub

---

**Note**: This project requires significant GPU resources for optimal performance. Minimum 16GB VRAM recommended for running multiple models simultaneously.
