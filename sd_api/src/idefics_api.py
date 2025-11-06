"""
To start the API navigate to the scripts folder and call:
gunicorn unidiffuser_api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
this requires a gunicorn to be installed (pip install should do the trick)
"""
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import torch
import torchvision.transforms as transforms
from diffusers import UniDiffuserPipeline
from fastapi import FastAPI, APIRouter, Response, UploadFile, File, Depends
from pydantic import BaseModel
from io import BytesIO
import numpy as np
from PIL import Image
import dotenv
from typing import List

dotenv.load_dotenv('.env')

class IdeficsModel:
    def __init__(self, model_id: str="HuggingFaceM4/idefics2-8b"):
        self.model_id = model_id
        self.set_model()
        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])

    def set_model(self, config=None):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
        ).to('cuda')
        torch.cuda.empty_cache()


    def talk_to_image(self, messages, images, image_size=None):
        if image_size is not None:
            image_size =  (512, 512)
        images = [Image.open(BytesIO(image)).convert("RGB").resize(image_size) for image in images]
        images = [self.transform(image) for image in images]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts



model = IdeficsModel()


app = FastAPI()
router = APIRouter()

class TalkToImage(BaseModel):
    messages: List[str]


class Image2Prompt(BaseModel):
    torch_dtype: str | None = 'float16'
    num_inference_steps: int | None = 20
    guidance_scale: float | None = 7.5

def _refactor_config(config: dict) -> dict:
    global model_config
    global prompt_config
    new_config = {'model_config': {},
                  'prompt_config': {}}
    for key, value in config.items():
        if key in model_config:
            new_config['model_config'][key] = value
        elif key in prompt_config:
            new_config['prompt_config'][key] = value
        else:
            new_config[key] = value
    return new_config

def get_bytes_value(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='png')
    return img_byte_arr.getvalue()
@router.post("/prompt2img",
             responses={
                 200: {
                     "content": {"image/png": {}}
                 }
             },
             response_class=Response
             )
async def get_image(item: Prompt2Image = Depends()):
    config = _refactor_config(vars(item))
    prompt = item.prompt
    del config['prompt']
    image = image_grid(model.gen_image(prompt, config))
    image = get_bytes_value(image)
    return Response(content = image, media_type='image/png')


@router.post("/img2prompt")
async def get_prompt(item: Image2Prompt = Depends(), image: UploadFile = File(None)):
    config = _refactor_config(vars(item))
    if image is not None:
        image = await image.read()
        prompt = model.gen_text(image, config)
    else:
        prompt = 'No image was passed!'
    return prompt

@router.post("/img2embed")
async def get_embed(image: UploadFile = File(None)):
    if image is not None:
        image = await image.read()
        embedding = model.gen_embed(image)
    else:
        embedding = 'No image was passed!'
    return embedding

app.include_router(router)