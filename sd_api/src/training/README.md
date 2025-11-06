# Training Scripts

Model fine-tuning and training utilities using Unsloth and other efficient training frameworks.

## Files

### `retrain_unsloth.py`
Script for fine-tuning language models using Unsloth's efficient training approach.

**Model:** google/gemma-2b (configurable)
**Task:** Image prompt generation for Stable Diffusion

#### Features:
- **4-bit quantization** for memory efficiency
- **PEFT/LoRA** training (Parameter-Efficient Fine-Tuning)
- **Optimized for long context** (2048 tokens)
- **Unsloth optimizations** - 30% less VRAM, 2x larger batch sizes
- **Gradient checkpointing** for memory savings
- **8-bit Adam optimizer** for reduced memory

#### Configuration

**Model Settings:**
```python
max_seq_length = 2048      # Context window
dtype = None               # Auto-detect (Float16/Bfloat16)
load_in_4bit = True        # Enable 4-bit quantization
```

**LoRA Settings:**
```python
r = 16                     # LoRA rank (8/16/32/64/128)
lora_alpha = 16           # LoRA scaling factor
lora_dropout = 0          # Dropout (0 is optimized)
target_modules = [        # Modules to apply LoRA
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

**Training Settings:**
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4    # Effective batch size = 2*4 = 8
warmup_steps = 5
max_steps = 60
learning_rate = 2e-4
optim = "adamw_8bit"              # Memory-efficient optimizer
```

#### Usage

**Basic Training:**
```bash
cd src/training
python retrain_unsloth.py
```

**Custom Model:**
```python
# Edit line 8-9
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.3-8B",  # Change this
    max_seq_length=4096,                    # Increase if needed
    # ...
)
```

**Custom Dataset:**
```python
# Edit lines 88-90
dataset = load_dataset("your-username/your-dataset", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

## Task: Image Prompt Generation

This script fine-tunes models to convert simple text into detailed Stable Diffusion prompts.

**Input Example:**
```
"A beautiful Sunday"
```

**Output Example:**
```
Description: A (serene cityscape:1.3) on a bright Sunday morning, with people
leisurely walking their dogs and street cafes bustling with early risers enjoying
breakfast. The style is reminiscent of a vibrant watercolor painting.
```

### Prompt Template

The script uses an Alpaca-style prompt:
```
### Instruction:
[Task description for image prompt generation]

### Input:
{user_text}

### Response:
{detailed_prompt}
```

## Hardware Requirements

**Minimum (gemma-2b):**
- GPU: 8GB VRAM (RTX 3070, RTX 4060 Ti)
- RAM: 16GB
- Storage: 20GB

**Recommended (7B models):**
- GPU: 16GB VRAM (RTX 4080, A4000)
- RAM: 32GB
- Storage: 50GB

**Large Models (13B-70B):**
- GPU: 40GB+ VRAM (A100, H100)
- RAM: 64GB+
- Multi-GPU recommended

## Unsloth Optimization Benefits

Traditional training vs Unsloth:
- **30% less VRAM** usage
- **2x larger batch sizes** possible
- **1.5-2x faster** training
- **Same final model quality**

Example memory usage for Llama-2-7b:
- Standard: 28GB VRAM
- Unsloth: 18GB VRAM ✅

## Training Best Practices

### 1. Start Small
```python
# Test with a small model first
model_name = "google/gemma-2b"  # 2B parameters
max_steps = 10  # Quick test
```

### 2. Monitor Loss
```python
# Add logging callback
from transformers import TrainerCallback

class LossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: Loss = {logs.get('loss', 0):.4f}")

trainer = SFTTrainer(
    # ...
    callbacks=[LossCallback()]
)
```

### 3. Save Checkpoints
```python
args=TrainingArguments(
    # ...
    output_dir="outputs",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,  # Keep only last 3 checkpoints
)
```

### 4. Evaluate Regularly
```python
# Add validation set
dataset_train = dataset.train_test_split(test_size=0.1)

trainer = SFTTrainer(
    # ...
    train_dataset=dataset_train["train"],
    eval_dataset=dataset_train["test"],
)
```

## Dataset Format

Expected format:
```python
{
    "input": "Simple text description",
    "output": "Detailed Stable Diffusion prompt with (emphasis:1.3) and style"
}
```

Example dataset entry:
```json
{
    "input": "A cat",
    "output": "Description: A (fluffy orange tabby cat:1.3) sitting on a windowsill, sunlight streaming through, watercolor style, soft pastel colors"
}
```

## After Training

### Save the Model
```python
# Save LoRA adapters (small, ~50MB)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Or save merged model (full size)
model.save_pretrained_merged("merged_model", tokenizer)
```

### Inference
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # Your trained model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable inference mode

prompt = alpaca_prompt.format(text="A sunset", response="")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## Hyperparameter Tuning

### Learning Rate
```python
# Too high: training unstable, poor quality
learning_rate = 2e-3  # ❌ Too high

# Good range for LoRA
learning_rate = 2e-4  # ✅ Recommended
learning_rate = 1e-4  # ✅ Conservative

# Too low: very slow learning
learning_rate = 1e-5  # ❌ Too low
```

### LoRA Rank
```python
# Higher rank = more parameters = better fit but more memory
r = 8   # Fast, small, might underfit
r = 16  # ✅ Balanced (recommended)
r = 32  # Better quality, more memory
r = 64  # High quality, 2x memory
r = 128 # Very high quality, 4x memory
```

### Batch Size
```python
# Effective batch size = per_device_batch * accumulation_steps * num_gpus
per_device_train_batch_size = 1   # Small GPU
gradient_accumulation_steps = 8   # Effective: 8

per_device_train_batch_size = 4   # Large GPU
gradient_accumulation_steps = 2   # Effective: 8

# Generally aim for effective batch size of 8-32
```

## Common Issues

**Out of Memory:**
```python
# Solutions:
per_device_train_batch_size = 1        # Reduce batch size
gradient_accumulation_steps = 8        # Increase accumulation
max_seq_length = 1024                  # Reduce sequence length
use_gradient_checkpointing = "unsloth" # Already enabled
```

**Poor Quality Output:**
```python
# Solutions:
max_steps = 200              # Train longer
learning_rate = 1e-4         # Lower learning rate
r = 32                       # Increase LoRA rank
# Add more training data
# Clean/improve dataset quality
```

**Training Too Slow:**
```python
# Solutions:
per_device_train_batch_size = 4  # Increase batch size
use_gradient_checkpointing = False  # Faster but more memory
max_seq_length = 512          # Shorter sequences
# Use smaller model
# Use multiple GPUs
```

**Model Overfitting:**
```python
# Solutions:
lora_dropout = 0.1           # Add dropout
weight_decay = 0.01          # Already set
max_steps = 50               # Train less
# Add more diverse training data
# Use data augmentation
```

## Advanced Techniques

### Multi-GPU Training
```python
# Automatic with accelerate
accelerate launch retrain_unsloth.py
```

### Mixed Precision
```python
args=TrainingArguments(
    # ...
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),  # Better on A100+
)
```

### Custom Scheduler
```python
args=TrainingArguments(
    # ...
    lr_scheduler_type="cosine",  # "linear", "cosine", "polynomial"
    warmup_ratio=0.1,             # Warmup for 10% of training
)
```

## Monitoring Training

### TensorBoard
```python
args=TrainingArguments(
    # ...
    logging_dir="./logs",
    logging_steps=1,
)

# View in browser
# tensorboard --logdir=./logs
```

### Weights & Biases
```python
# pip install wandb
import wandb
wandb.init(project="prompt-generation")

args=TrainingArguments(
    # ...
    report_to="wandb",
)
```

## Model Zoo

Compatible models (must support LoRA):
- **Gemma 2B/7B** (Google)
- **Llama 3.1/3.3 8B/70B** (Meta)
- **Mistral 7B** (Mistral AI)
- **Qwen 2.5 7B/14B** (Alibaba)
- **Phi-3 Mini** (Microsoft)

Not all models supported - check [Unsloth docs](https://github.com/unslothai/unsloth).

## Dependencies

- **unsloth** >= 2025.11.1 - Efficient training
- **transformers** >= 4.57.1 - Model framework
- **trl** >= 0.13.0 - Reinforcement learning trainer
- **datasets** >= 3.2.0 - Dataset management
- **torch** >= 2.6.0 - Deep learning backend
- **peft** - Parameter-efficient fine-tuning (auto-installed)

## Future Improvements

- [ ] Add evaluation metrics (BLEU, ROUGE)
- [ ] Implement early stopping
- [ ] Add learning rate finder
- [ ] Support for instruction tuning
- [ ] Add DPO/RLHF training
- [ ] Implement multi-task learning
- [ ] Add automatic hyperparameter search
- [ ] Support for continuous pre-training
- [ ] Add model merging utilities
- [ ] Implement curriculum learning

## References

- [Unsloth](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Alpaca Format](https://github.com/tatsu-lab/stanford_alpaca)

## License

Check individual model licenses before commercial use.
