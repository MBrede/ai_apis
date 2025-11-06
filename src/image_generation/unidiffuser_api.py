"""
To start the API navigate to the scripts folder and call:
gunicorn unidiffuser_api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
this requires a gunicorn to be installed (pip install should do the trick)
"""

from io import BytesIO

import dotenv
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import UniDiffuserPipeline
from fastapi import APIRouter, Depends, FastAPI, File, Response, UploadFile
from PIL import Image
from pydantic import BaseModel

dotenv.load_dotenv(".env")


class UniDiffusionModel:
    def __init__(self, model_id: str = "thu-ml/unidiffuser-v1"):
        self.loaded_lora = None
        self.config = {"model_id": model_id, "model_config": {"torch_dtype": "float16"}}
        self.model_id = model_id
        self.set_model()
        self.transform = transforms.Compose([transforms.PILToTensor()])

    def _validate_model_config(self, config: dict) -> dict:
        if "torch_dtype" in config:
            config["torch_dtype"] = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }[config["torch_dtype"]]
        return config

    def _load_pipeline(self):
        pipe = UniDiffuserPipeline.from_pretrained(self.model_id, **self.config)
        self.pipe = pipe.to("cuda")
        torch.cuda.empty_cache()

    def set_model(self, config=None):
        reset = False
        if config is not None:
            if "model_config" in config:
                model_config = self._validate_model_config(config["model_config"])
                if any([self.config[k] != model_config[k] for k in model_config]):
                    self.config = model_config
                    reset = True
        else:
            if "model_config" in self.config:
                self.config = self._validate_model_config(self.config["model_config"])
            reset = True
        if reset:
            self._load_pipeline()
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False

    def _prompt_to_embedding(self, prompt):
        tokens = self.pipe.tokenizer(prompt, return_tensors="pt")
        prompt_embeds = self.pipe.text_encoder(
            tokens["input_ids"][:, 0 : self.pipe.tokenizer.model_max_length].to("cuda"),
            attention_mask=tokens["attention_mask"][:, 0 : self.pipe.tokenizer.model_max_length].to(
                "cuda"
            ),
        )
        prompt_embeds = prompt_embeds[0]
        embedded_text = self.pipe.tokenizer.batch_decode(
            tokens["input_ids"][:, 1 : (self.pipe.tokenizer.model_max_length - 1)]
        )
        return prompt_embeds.tolist(), embedded_text

    def gen_image(self, prompt, config):
        n = config["count_returned"]
        negative_prompt = config["prompt_config"]["negative_prompt"]
        del config["prompt_config"]["negative_prompt"]
        seed = config["seed"]
        args = {
            "prompt": [prompt] * n,
            "negative_prompt": [negative_prompt] * n,
            "generator": [torch.Generator("cuda").manual_seed(i) for i in range(seed, seed + n)],
        } | config["prompt_config"]
        args["width"] = int(config["width"] // 8 * 8)
        args["height"] = int(config["height"] // 8 * 8)
        self.set_model(config)
        sample = self.pipe(**args)
        return sample.images

    def gen_text(self, image, config):
        image = Image.open(BytesIO(image)).convert("RGB")
        image = image.resize((512, 512))
        self.set_model(config)
        sample = self.pipe(image=image)
        return sample.text[0]

    def gen_embed(self, image):
        image = Image.open(BytesIO(image)).convert("RGB")
        image = image.resize((512, 512))
        img_tensor = self.transform(image).to("cuda")
        img_tensor = img_tensor[None, :, :, :]
        clip_emb = self.pipe.encode_image_clip_latents(
            img_tensor, batch_size=1, num_prompts_per_image=1, dtype=torch.float16, device="cuda"
        )
        vae_emb = self.pipe.encode_image_vae_latents(
            img_tensor,
            batch_size=1,
            num_prompts_per_image=1,
            dtype=torch.float16,
            device="cuda",
            do_classifier_free_guidance=False,
        )
        return {"CLIP_embeddings": clip_emb[0].tolist(), "vae_emb": vae_emb[0].tolist()}


def image_grid(imgs):
    rows = int(np.ceil(len(imgs) / 4))
    cols = int(np.ceil(len(imgs) / rows))
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


model = UniDiffusionModel()
model_config = ["torch_dtype"]
prompt_config = ["num_inference_steps", "guidance_scale", "negative_prompt"]

app = FastAPI()
router = APIRouter()


class Prompt2Image(BaseModel):
    prompt: str | None = "Ein schöner Strand im Süden Schleswig-Holsteins, digitales Kunstwerk"
    torch_dtype: str | None = "float16"
    num_inference_steps: int | None = 20
    count_returned: int | None = 1
    seed: int | None = 0
    guidance_scale: float | None = 7.5
    width: int | None = 512
    height: int | None = 512
    negative_prompt: str | None = "blurry, low resolution, low quality"


class Image2Prompt(BaseModel):
    torch_dtype: str | None = "float16"
    num_inference_steps: int | None = 20
    guidance_scale: float | None = 7.5


def _refactor_config(config: dict) -> dict:
    global model_config
    global prompt_config
    new_config = {"model_config": {}, "prompt_config": {}}
    for key, value in config.items():
        if key in model_config:
            new_config["model_config"][key] = value
        elif key in prompt_config:
            new_config["prompt_config"][key] = value
        else:
            new_config[key] = value
    return new_config


def get_bytes_value(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="png")
    return img_byte_arr.getvalue()


@router.post(
    "/prompt2img", responses={200: {"content": {"image/png": {}}}}, response_class=Response
)
async def get_image(item: Prompt2Image = Depends()):
    config = _refactor_config(vars(item))
    prompt = item.prompt
    del config["prompt"]
    image = image_grid(model.gen_image(prompt, config))
    image = get_bytes_value(image)
    return Response(content=image, media_type="image/png")


@router.post("/img2prompt")
async def get_prompt(item: Image2Prompt = Depends(), image: UploadFile = File(None)):
    config = _refactor_config(vars(item))
    if image is not None:
        image = await image.read()
        prompt = model.gen_text(image, config)
    else:
        prompt = "No image was passed!"
    return prompt


@router.post("/img2embed")
async def get_embed(image: UploadFile = File(None)):
    if image is not None:
        image = await image.read()
        embedding = model.gen_embed(image)
    else:
        embedding = "No image was passed!"
    return embedding


app.include_router(router)
