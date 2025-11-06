# Text Analysis APIs

Sentiment analysis and text classification using HuggingFace transformers and SetFit.

## Files

### `sentiment_api.py`
**Model:** Configurable (default: oliverguhr/german-sentiment-bert)
**GPU:** Optional (uses available device)

General-purpose sentiment analysis API supporting any HuggingFace sentiment model.

#### Features:
- Dynamic model loading
- Multi-text batch processing
- Support for any HuggingFace classification model
- Probability scores for all labels

#### Endpoints:
- `POST /predict_sentiment/` - Analyze sentiment
  - Request body: `{"text": ["text1", "text2", ...], "model": "model_name"}`
  - Response: `{"answer": [{label: probability}, ...]}`

#### Usage Examples:

**German Sentiment Analysis:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict_sentiment/",
    json={
        "text": [
            "Das ist fantastisch!",
            "Ich bin sehr enttäuscht.",
            "Es war okay."
        ],
        "model": "oliverguhr/german-sentiment-bert"
    }
)

results = response.json()["answer"]
for text, sentiment in zip(texts, results):
    print(f"{text} → {sentiment}")
# Output:
# Das ist fantastisch! → {'positive': 0.98, 'negative': 0.01, 'neutral': 0.01}
# Ich bin sehr enttäuscht. → {'positive': 0.02, 'negative': 0.95, 'neutral': 0.03}
# Es war okay. → {'positive': 0.15, 'negative': 0.10, 'neutral': 0.75}
```

**English Sentiment:**
```python
response = requests.post(
    "http://localhost:8000/predict_sentiment/",
    json={
        "text": ["This is amazing!", "I hate this."],
        "model": "distilbert-base-uncased-finetuned-sst-2-english"
    }
)
```

**Popular Models:**
- `oliverguhr/german-sentiment-bert` - German sentiment (3 classes)
- `distilbert-base-uncased-finetuned-sst-2-english` - English binary sentiment
- `cardiffnlp/twitter-roberta-base-sentiment` - Twitter sentiment
- `nlptown/bert-base-multilingual-uncased-sentiment` - Multi-language (5 stars)

---

### `text_classification_api.py`
**Model:** Configurable (default: oliverguhr/german-sentiment-bert)
**GPU:** Optional (uses available device)

Advanced text classification supporting both standard transformers and SetFit models.

#### Features:
- Automatic model type detection (SetFit vs standard)
- Dynamic model loading based on HuggingFace API
- Batch processing
- Support for few-shot learning (SetFit)

#### Endpoints:
- `POST /predict_proba/` - Classify text with probabilities
  - Request body: `{"text": ["text1", "text2"], "model": "model_name"}`
  - Response: `{"answer": [{label: probability}, ...]}`

#### Usage Examples:

**Standard Classification:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict_proba/",
    json={
        "text": ["This product is great!", "Terrible experience"],
        "model": "distilbert-base-uncased-finetuned-sst-2-english"
    }
)

predictions = response.json()["answer"]
```

**SetFit (Few-Shot Learning):**
```python
# SetFit models are efficient few-shot classifiers
response = requests.post(
    "http://localhost:8000/predict_proba/",
    json={
        "text": ["I love this!", "Not good at all"],
        "model": "SetFit/distilbert-base-uncased__sst2__train-32-0"
    }
)
```

**Topic Classification:**
```python
response = requests.post(
    "http://localhost:8000/predict_proba/",
    json={
        "text": [
            "The stock market crashed today",
            "Scientists discover new planet",
            "Team wins championship"
        ],
        "model": "facebook/bart-large-mnli"
    }
)
```

## Model Types

### Standard Transformers
- Full transformer models
- High accuracy
- More GPU memory
- Slower inference
- Examples: BERT, RoBERTa, DistilBERT

### SetFit Models
- Few-shot learning
- Very efficient
- Smaller memory footprint
- Faster inference
- Great for limited training data
- Examples: SetFit/[model_name]

## Performance Comparison

| Model Type | Accuracy | Speed | Memory | Training Data |
|------------|----------|-------|--------|---------------|
| BERT-base | Highest | Slow | 500MB | 10k+ samples |
| DistilBERT | High | Medium | 250MB | 5k+ samples |
| SetFit | High | Fast | 100MB | 8-32 samples |

## Supported Tasks

1. **Sentiment Analysis**
   - Binary (positive/negative)
   - Multi-class (positive/neutral/negative)
   - Fine-grained (1-5 stars)

2. **Topic Classification**
   - News categories
   - Product categories
   - Intent detection

3. **Toxicity Detection**
   - Hate speech
   - Offensive language
   - Cyberbullying

4. **Emotion Detection**
   - Joy, sadness, anger, fear, etc.

5. **Language Detection**
   - Identify language of text

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (speeds up by 5-10x)

**Recommended:**
- GPU: 4GB+ VRAM
- RAM: 16GB
- CPU: 8+ cores

**CPU-only:** Both APIs work fine on CPU for moderate loads.

## Optimization Tips

1. **Batch processing:**
   ```python
   # Process multiple texts at once
   response = requests.post(..., json={"text": texts[:100]})
   ```

2. **Model caching:**
   The API loads models on first use and keeps them in memory.

3. **Use DistilBERT:**
   40% faster and 40% smaller than BERT with 97% of the performance

4. **SetFit for few-shot:**
   When you have limited training data, SetFit is much more efficient

## Differences Between APIs

| Feature | sentiment_api.py | text_classification_api.py |
|---------|-----------------|----------------------------|
| SetFit support | ❌ No | ✅ Yes |
| Auto model detection | ❌ No | ✅ Yes (via HuggingFace API) |
| HuggingFace API calls | ❌ No | ✅ Yes |
| .env required | ❌ No | ✅ Yes (for hf_token) |

**Recommendation:** Use `text_classification_api.py` for production - it's more feature-complete.

## Known Issues

1. **Global model cache** - Models not unloaded between requests
2. **No error handling** for invalid model names
3. **No timeout** for model downloads
4. **Memory leaks** possible with frequent model switching
5. **No rate limiting**
6. **sentiment_api.py** doesn't require authentication (should use hf_token)

## Usage Patterns

### Single Model (Fast):
```python
# Load once, use many times
for text_batch in batches:
    result = requests.post(..., json={"text": text_batch, "model": "same-model"})
```

### Multiple Models (Slow):
```python
# Switching models requires reloading
for model in models:
    result = requests.post(..., json={"text": texts, "model": model})
    # Model loading time on each iteration
```

## Environment Variables

**text_classification_api.py only:**
- `hf_token` - HuggingFace API token (for model info queries)

Create `.env` file:
```bash
hf_token=hf_your_token_here
```

## Troubleshooting

**Model not found:**
- Check model name on HuggingFace
- Ensure model is a classification model
- Try different model names

**Out of memory:**
- Use smaller model (DistilBERT instead of BERT)
- Reduce batch size
- Use CPU instead of GPU

**Slow inference:**
- Use GPU if available
- Switch to distilled models
- Reduce input text length (truncate at 512 tokens)

**SetFit model fails:**
- Ensure you're using `text_classification_api.py`
- Check SetFit is installed: `pip install setfit`

## Future Improvements

- [ ] Add model warmup on startup
- [ ] Implement model unloading after timeout
- [ ] Add authentication and rate limiting
- [ ] Support for zero-shot classification
- [ ] Add confidence thresholds
- [ ] Implement result caching
- [ ] Add batch size limits
- [ ] Better error messages
- [ ] Add text preprocessing options
- [ ] Support for multi-label classification
- [ ] Add explain ability (attention weights)
- [ ] Implement A/B testing
- [ ] Add model performance metrics endpoint

## Popular Pre-trained Models

### Sentiment Analysis:
```python
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",  # English binary
    "cardiffnlp/twitter-roberta-base-sentiment",        # Twitter
    "nlptown/bert-base-multilingual-uncased-sentiment", # Multi-lang
    "oliverguhr/german-sentiment-bert",                 # German
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",  # 6 languages
]
```

### Topic Classification:
```python
models = [
    "facebook/bart-large-mnli",           # Zero-shot
    "joeddav/xlm-roberta-large-xnli",    # Multi-language
    "typeform/distilbert-base-uncased-mnli",  # Fast
]
```

### Toxicity Detection:
```python
models = [
    "martin-ha/toxic-comment-model",
    "unitary/toxic-bert",
    "facebook/roberta-hate-speech-dynabench-r4-target",
]
```

## Example Integration

```python
class TextClassifier:
    """Client for text classification API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def predict(
        self,
        texts: list[str],
        model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> list[dict]:
        """Classify texts with probabilities."""
        response = self.session.post(
            f"{self.base_url}/predict_proba/",
            json={"text": texts, "model": model}
        )
        response.raise_for_status()
        return response.json()["answer"]

    def sentiment(self, texts: list[str]) -> list[str]:
        """Get sentiment labels (positive/negative)."""
        results = self.predict(texts)
        return [max(r, key=r.get) for r in results]

# Usage
classifier = TextClassifier()
labels = classifier.sentiment([
    "This is awesome!",
    "I hate this",
    "It's okay"
])
print(labels)  # ['positive', 'negative', 'neutral']
```

## Dependencies

- **transformers** >= 4.57.1 - HuggingFace models
- **torch** >= 2.6.0 - Deep learning backend
- **setfit** >= 1.1.3 - Few-shot learning (text_classification_api only)
- **huggingface-hub** >= 0.26.5 - Model hub access
- **fastapi** >= 0.115.8 - API framework
- **pydantic** >= 2.10.5 - Request/response validation

## References

- [HuggingFace Models](https://huggingface.co/models?pipeline_tag=text-classification)
- [SetFit](https://github.com/huggingface/setfit)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
