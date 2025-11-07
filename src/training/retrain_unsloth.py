import torch
from unsloth import FastLanguageModel

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2b",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction: 
Your task is to create a vivid and detailed image description suitable for generating images via a 
diffusion model. Start by ensuring any non-English text is translated into English. Each description should begin 
with the keyword 'Description:', followed by a detailed depiction that includes specific elements such as setting, 
mood, key objects, and any notable art styles. For instance, for the phrase: 'Ein schöner Sonntag' translated as 'A 
beautiful Sunday', you might write: 'Description: A (serene cityscape:1.3) on a bright Sunday morning, with people 
leisurely walking their dogs and street cafes bustling with early risers enjoying breakfast. The style is reminiscent 
of a vibrant watercolor painting.' \nHere are more examples to guide you: \n 'A stormy sea' - 'Description: A (
tumultuous ocean scene:1.3) with high waves crashing under a dark, brooding sky. The art style is dynamic and 
impressionistic, capturing the raw energy of nature.' \n 'Quiet library' - 'Description: An image of an (old and vast 
library:1.3) with rows of wooden bookshelves, soft light filtering through stained glass windows, creating a tranquil 
and scholarly ambiance. The style is photorealistic with attention to texture and lighting.' \n 'A serene old town in 
watercolor' - 'Description: A (serene old town:1.3) in Europe, rendered in the architecture watercolor style. 
Cobblestone streets, quaint cafes, and blooming flowers in window sills evoke warmth and history. The art style 
emphasizes soft watercolor tones and light brush strokes that highlight architectural details and vibrant town life.' 
\n 'A futuristic cyberpunk cityscape' - 'Description: A (futuristic city:1.3) at night, illuminated by neon lights 
and digital billboards, showcasing a stark contrast between the dark sky and vivid cityscape colors. The scene 
includes advanced architecture with skyscrapers and flying vehicles, capturing the high-tech, dystopian world in a 
cyberpunk style.' \n 'A traditional Japanese garden in autumn hues' - 'Description: A (traditional Japanese 
garden:1.3) during autumn, portrayed in blacks and reds. The scene features a tranquil pond, a wooden bridge, 
and maple trees with fiery red leaves. The artwork reflects Japanese aesthetic balance and harmony, using a dominant 
black and red watercolor palette.' \n 'A heroic worker in vintage Soviet matchbox style' - 'Description: An image 
styled like vintage Soviet matchboxes, featuring a (heroic worker:1.3) against an industrial backdrop. Bold red and 
yellow colors dominate, echoing Soviet-era propaganda style. The composition is simple yet powerful, reminiscent of 
graphic art on mid-20th-century Soviet matchboxes.' \n 'An elegant woman in Art Nouveau style' - 'Description: An 
image of a (woman in an elegant, flowing gown:1.3), surrounded by flowers and vines in an Art Nouveau style. The 
artwork features fluid, ornate lines and organic forms that integrate into her attire and the surroundings, 
with a soft yet vibrant palette typical of Art Nouveau.' \n\n You can weigh parts of the description using this 
syntax '(text-snippet:weight)' where a weight like 1.3 is quite high. Make sure that the most important parts of the 
text are emphasized like in the examples with emphasis. Based on the given text '{text}', create a similarly detailed 
image description. 

### Input:
{text}

### Response:
{response}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(text=input, response=output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


pass

from datasets import load_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
