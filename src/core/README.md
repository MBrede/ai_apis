# Core Utilities

Shared infrastructure for all APIs.

## Modules

- `auth.py` - API key authentication (supports multiple keys)
- `config.py` - Centralized configuration
- `buffer_class.py` - Automatic GPU memory management
- `database.py` - MongoDB connection utilities
- `bot.py` - Telegram bot interface

## Buffer Class

Automatic model unloading after inactivity:

```python
from src.core.buffer_class import Model_Buffer

class MyModelBuffer(Model_Buffer):
    def load_model(self, timeout=300):
        super().load_model(timeout=timeout)
        self.model = load_your_model()
        self.loaded_at = datetime.now()
        if self.timer:
            self.timer.start()

buffer = MyModelBuffer()
buffer.load_model(timeout=600)  # 10 minutes
```
