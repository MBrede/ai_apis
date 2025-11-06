# Core Utilities

Core components for API infrastructure and user interaction.

## Files

### `buffer_class.py`
Abstract base class for managing ML model loading/unloading with automatic memory cleanup.

**Features:**
- Automatic model unloading after timeout
- GPU memory management (CUDA cache clearing)
- Garbage collection integration
- Thread-safe timer management

**Usage:**
```python
from buffer_class import Model_Buffer

class MyModel(Model_Buffer):
    def load_model(self, model_path, timeout=300, **kwargs):
        super().load_model(timeout=timeout)  # Set up timer
        self.model = load_my_model(model_path)
        self.timer.start()  # Start auto-unload timer
```

**Fixed Issues:**
- ✅ Added missing `import gc`
- ✅ Fixed `timer` → `self.timer` reference bug

### `api_request.py`
Client utility functions for making requests to the AI APIs.

**Functions:**
- `api_request()` - Generate images via Stable Diffusion API
- `data_request()` - Generate training data using LLM

**⚠️ Issues:**
- Hardcoded IP addresses (should use environment variables)
- No retry logic for failed requests
- Limited error handling

**Usage:**
```python
from api_request import api_request

# Text to image
image = api_request(
    prompt="A beautiful landscape",
    model_id="stabilityai/stable-diffusion-2-1",
    num_inference_steps=30,
    guidance_scale=7.5
)

# Image to image
image = api_request(
    image_path="input.jpg",
    prompt="Make it sunset",
    num_inference_steps=20
)
```

### `bot.py`
Telegram bot interface for AI services.

**Features:**
- Image generation from text prompts
- Image-to-image transformation
- LLM chat mode
- Audio transcription
- User management and admin controls
- Per-user settings and preferences

**Commands:**

**User Commands:**
- `/start` - Initialize bot
- `/help` - Show all commands
- `/llm` - Toggle between SD and LLM mode
- `/get_parameters` - View current settings
- `/get_loras` - List available LORA models
- `/get_sd` - List Stable Diffusion models
- `/set_parameters key=value` - Update settings
- `/assist <text>` - Get LLM help with prompts
- `/assist_create <text>` - Generate prompt + image
- Send image with caption → Image-to-image generation
- Send audio → Transcribe with Whisper

**Admin Commands:**
- `/add_user <user_id>` - Add user to whitelist
- `/del_user <user_id>` - Remove user
- `/add_admin <user_id>` - Grant admin privileges
- `/remove_admin <user_id>` - Revoke admin privileges
- `/list_users` - Show all users
- `/list_contacts` - Show contact attempts
- `/add_lora <name>` - Add new LORA model

**Configuration:**

User data stored in:
- `users.json` - User settings and permissions
- `contacts.json` - Unauthorized contact attempts

**Environment Variables:**
- `telegram_token` - Telegram Bot API token

**Start Command:**
```bash
cd src/core
python bot.py
```

**⚠️ Security Issues:**
- Hardcoded API endpoints (should use environment)
- No rate limiting per user
- User IDs stored in plain JSON (consider database)
- No encryption for sensitive data

**Usage Example:**

1. Get bot token from [@BotFather](https://t.me/botfather)
2. Add to `.env`: `telegram_token=YOUR_TOKEN_HERE`
3. Run bot: `python bot.py`
4. Add your Telegram user ID as admin in `users.json`:
```json
{
  "YOUR_USER_ID": {
    "admin": true,
    "mode": "sd",
    "current_settings": {
      "model_id": "runwayml/stable-diffusion-v1-5",
      "torch_dtype": "float16",
      "num_inference_steps": 20,
      "count_returned": 1,
      "seed": 0,
      "guidance_scale": 7.5,
      "negative_prompt": "blurry, low resolution, low quality",
      "width": 512,
      "height": 512,
      "lora": ""
    }
  }
}
```

## Architecture

```
┌─────────────┐
│ Telegram    │
│ User        │
└──────┬──────┘
       │
       v
┌──────────────────┐
│  bot.py          │
│  - User auth     │
│  - Command parse │
│  - State mgmt    │
└────┬─────┬───────┘
     │     │
     │     └──────────────┐
     │                    │
     v                    v
┌────────────┐    ┌──────────────┐
│ SD API     │    │  LLM API     │
│ (port 1234)│    │  (port 8000) │
└────────────┘    └──────────────┘
     │                    │
     v                    v
┌────────────┐    ┌──────────────┐
│ Whisper API│    │ Other APIs   │
│ (port 8080)│    │              │
└────────────┘    └──────────────┘
```

## Dependencies

- **fastapi** - API framework (for api_request.py)
- **python-telegram-bot** - Telegram bot SDK
- **python-dotenv** - Environment variable management
- **torch** - PyTorch (for buffer_class.py)
- **requests** - HTTP client
- **Pillow** - Image processing

## Future Improvements

### buffer_class.py
- [ ] Add async support for non-blocking operations
- [ ] Implement model warming (keep frequently used models loaded)
- [ ] Add memory usage monitoring
- [ ] Support multiple GPU management

### api_request.py
- [ ] Move hardcoded IPs to environment variables
- [ ] Add retry logic with exponential backoff
- [ ] Implement connection pooling
- [ ] Add request/response logging
- [ ] Better error messages
- [ ] Add timeout configuration

### bot.py
- [ ] Add database for user management (SQLite/PostgreSQL)
- [ ] Implement rate limiting per user
- [ ] Add usage statistics and quotas
- [ ] Implement payment/subscription system
- [ ] Add image gallery and history
- [ ] Support for multiple languages (i18n)
- [ ] Add inline query support
- [ ] Implement webhook mode (instead of polling)
- [ ] Add queue system for heavy operations
- [ ] Better error recovery and user feedback

## Testing

```bash
# Test buffer class
pytest tests/test_buffer_class.py

# Test API client
pytest tests/test_api_request.py

# Test bot (requires mock Telegram API)
pytest tests/test_bot.py
```

## Known Issues

1. ✅ **Fixed**: `buffer_class.py` - missing `import gc` and `timer` variable bug
2. **Hardcoded endpoints** in all files - need environment variables
3. **No authentication** in api_request.py
4. **Plain text storage** of user data in bot.py
5. **No input validation** in bot commands
6. **Limited error handling** throughout
