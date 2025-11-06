# Text Classification API

Text classification and sentiment analysis.

## Endpoints

- `POST /predict_proba/` - Get prediction probabilities
- `GET /buffer_status` - Check model status

## Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict_proba/",
    headers={"X-API-Key": "your-key"},
    json={
        "texts": ["This is amazing!", "This is terrible."],
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment"
    }
)
```

## Features

- Automatic GPU memory management (5-minute timeout)
- Supports both SetFit and transformer models
- Any HuggingFace classification model
- Batch processing support
