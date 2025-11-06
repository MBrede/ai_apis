"""
gunicorn huggingface_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 -t 30000
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer, BitsAndBytesConfig
from typing import Union
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import torch
from dotenv import load_dotenv
import os

load_dotenv()

class LLM:

    def __init__(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                             low_cpu_mem_usage=True,
                                                             quantization_config=bnb_config,
                                                             token=os.environ["hf_token"],
                                                             device_map="cuda:1"
                                                             )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  token=os.environ["hf_token"])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.generation_config = GenerationConfig(max_new_tokens=500,
                                             temperature=0.4,
                                             top_p=0.95,
                                             top_k=40,
                                             repetition_penalty=1.2,
                                             bos_token_id=self.tokenizer.bos_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id,
                                             tensor_parallel_size=4,
                                             do_sample=True,
                                             use_cache=True,
                                             output_attentions=False,
                                             output_hidden_states=False,
                                             output_scores=False,
                                             remove_invalid_values=True
                                             )
        self.streamer = TextStreamer(self.tokenizer)

        self.prompt_template = (
            "Your task is to create a vivid and detailed image description suitable for generating images via a diffusion model. "
            "Start by ensuring any non-English text is translated into English. Each description should begin with the keyword 'Description:', followed by a detailed depiction that includes specific elements such as setting, mood, key objects, and any notable art styles. "
            "For instance, for the phrase: 'Ein schöner Sonntag' translated as 'A beautiful Sunday', you might write: "
            "'Description: A (serene cityscape:1.3) on a bright Sunday morning, with people leisurely walking their dogs and street cafes bustling with early risers enjoying breakfast. The style is reminiscent of a vibrant watercolor painting.' "
            "\nHere are more examples to guide you: \n"
            "'A stormy sea' - 'Description: A (tumultuous ocean scene:1.3) with high waves crashing under a dark, brooding sky. The art style is dynamic and impressionistic, capturing the raw energy of nature.' \n"
            "'Quiet library' - 'Description: An image of an (old and vast library:1.3) with rows of wooden bookshelves, soft light filtering through stained glass windows, creating a tranquil and scholarly ambiance. The style is photorealistic with attention to texture and lighting.' \n"
            "'A serene old town in watercolor' - 'Description: A (serene old town:1.3) in Europe, rendered in the architecture watercolor style. Cobblestone streets, quaint cafes, and blooming flowers in window sills evoke warmth and history. The art style emphasizes soft watercolor tones and light brush strokes that highlight architectural details and vibrant town life.' \n"
            "'A futuristic cyberpunk cityscape' - 'Description: A (futuristic city:1.3) at night, illuminated by neon lights and digital billboards, showcasing a stark contrast between the dark sky and vivid cityscape colors. The scene includes advanced architecture with skyscrapers and flying vehicles, capturing the high-tech, dystopian world in a cyberpunk style.' \n"
            "'A traditional Japanese garden in autumn hues' - 'Description: A (traditional Japanese garden:1.3) during autumn, portrayed in blacks and reds. The scene features a tranquil pond, a wooden bridge, and maple trees with fiery red leaves. The artwork reflects Japanese aesthetic balance and harmony, using a dominant black and red watercolor palette.' \n"
            "'A heroic worker in vintage Soviet matchbox style' - 'Description: An image styled like vintage Soviet matchboxes, featuring a (heroic worker:1.3) against an industrial backdrop. Bold red and yellow colors dominate, echoing Soviet-era propaganda style. The composition is simple yet powerful, reminiscent of graphic art on mid-20th-century Soviet matchboxes.' \n"
            "'An elegant woman in Art Nouveau style' - 'Description: An image of a (woman in an elegant, flowing gown:1.3), surrounded by flowers and vines in an Art Nouveau style. The artwork features fluid, ornate lines and organic forms that integrate into her attire and the surroundings, with a soft yet vibrant palette typical of Art Nouveau.' \n\n"
            "You can weigh parts of the description using this syntax '(text-snippet:weight)' where a weight like 1.3 is quite high. "
            "Make sure that the most important parts of the text are emphasized like in the examples with emphasis."
            "Based on the given text '{text}', create a similarly detailed image description."
        )

    def __call__(self, text):
        messages = {"role": "user", "prompt": self.prompt_template.format(text=text)}
        return self.generate_naked(**messages)

    def generate_naked(self, prompt, role: str = None, temp: float = 0.3, max_new_tokens=500):
        prompt = f"{role} USER: {prompt} ASSISTANT:"
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.generation_config.temperature = temp
        self.generation_config.max_tokens = max_new_tokens
        output_tokens = self.model.generate(**input_tokens, generation_config=self.generation_config,
                                            streamer=self.streamer)[0]
        answer = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return answer.split('ASSISTANT:')[1]


llm = LLM()
app = FastAPI()
router = APIRouter()


@router.get("/llm_prompt_assistance")
async def llm_prompt_assistance(text: str):
    return llm(text)


@router.get("/llm_interface")
async def llm_interface(text: str, role: str = None, temp: float = .3):
    return llm.generate_naked(prompt=text, role=role, temp=temp)


class Item(BaseModel):
    system: str | None = 'Du bist ein hilfreicher Assistent.'
    messages: str | None = 'Erzähl mir eine Geschichte über Schnabeltiere und Data Science.'
    temperature: float | None = 0.1
    max_tokens: int | None = 500


@router.post("/answer/")
def get_answer(item: Item):
    answer = llm.generate_naked(prompt=item.message, role=item.system, temp=item.temperature, max_new_tokens=item.max_tokens)
    return {"message": answer}


app.include_router(router)
