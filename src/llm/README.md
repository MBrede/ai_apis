# LLM APIs (Legacy)

Legacy LLM endpoints - kept for backward compatibility.

**Note**: For current LLM access, use OLLAMA endpoint instead.

## Endpoints

- `POST /llm_answer/` - Get LLM response
- `POST /json_answer/` - Structured JSON response
- `GET /list_available_llms` - List online models

## Recommendation

Use the OLLAMA endpoint configured in `.env`:
```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=2345
OLLAMA_MODEL=llama3.3
```
