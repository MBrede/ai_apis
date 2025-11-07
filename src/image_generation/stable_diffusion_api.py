"""
Stable Diffusion API with LORA support and authentication.

To start the API:
    gunicorn stable_diffusion_api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:1234
"""

import json
import os
import re
import subprocess
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import torch
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    FluxPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from fastapi import APIRouter, Depends, FastAPI, File, Response, UploadFile
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

from src.core.auth import verify_admin_key, verify_api_key
from src.core.buffer_class import Model_Buffer
from src.core.config import config

sd_paths = {
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
    "SD 2.0": "stabilityai/stable-diffusion-2",
    "SD 2.1": "stabilityai/stable-diffusion-2-1",
    "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "Flux.1 D": "black-forest-labs/FLUX.1-dev",
}


def prep_name(string):
    string = string.encode("ascii", "ignore").decode("ascii")
    string = re.sub(r"\s", "_", string.lower())
    return re.sub(r"[^a-z_0-9]", "", string.lower())


class DiffusionModel(Model_Buffer):

    schedulers = {"euler": EulerDiscreteScheduler}

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", timeout: int = 600):
        # Initialize buffer (sets self.model, self.pipeline, etc. to None)
        super().__init__()

        self.loaded_lora = None
        self.config = {"torch_dtype": torch.float16}
        self.type = "prompt2img"
        self.model_id = None
        self.loaded_pipeline = None

        # Load model with timeout (10 minutes default for slow SD models)
        self.load_model(model_id=model_id, timeout=timeout)
        self.load_implemented_loras()

    def load_model(self, model_id: str = None, timeout: int = 600, **kwargs):
        """Load the diffusion model with automatic unloading after timeout."""
        # Set up timer using parent's load_model
        super().load_model(timeout=timeout)

        # Load the pipeline
        if model_id:
            self.model_id = model_id

        self.set_model({"model_id": self.model_id, "type": "prompt2img"})
        self.loaded_at = datetime.now()

        # Start timer if configured
        if self.timer:
            self.timer.start()

    def add_new_lora(self, lora_name):
        search_progress = ""
        if lora_name not in self.implemeted_loras:
            self.implemeted_loras[lora_name] = {}
            if lora_name.isdigit():
                self.implemeted_loras[lora_name]["model_id"] = int(lora_name)
            search_progress = self._update_lora_info()
        return search_progress

    def _update_lora_info(self):
        updated_info = {}
        ret = ""
        for model in tqdm(self.implemeted_loras, "updating info"):
            if "base_model" not in self.implemeted_loras[model]:
                if "model_id" in self.implemeted_loras[model]:
                    id = self.implemeted_loras[model]["model_id"]
                    answer = requests.get(f"https://civitai.com/api/v1/models/{id}")
                else:
                    answer = requests.get(
                        f'https://civitai.com/api/v1/models?query={model}&token={os.environ["civit_key"]}&nsfw=false'
                    )
                if answer.ok:
                    try:
                        info = json.loads(answer.content.decode("utf-8"))["items"]
                        ret += "\n" + f"searchterm {model} lead to {len(info)} models"
                    except KeyError:
                        info = [json.loads(answer.content.decode("utf-8"))]
                    try:
                        if len(info) > 0:
                            info = info[0]
                            info["name"] = prep_name(info["name"])
                            ret += "\n" + f',collecting the first one: {info["name"]}'
                            file_info = info["modelVersions"][0]["files"][0]
                            updated_info[info["name"]] = {
                                "model_id": info["id"],
                                "allow_no_mention": info["allowNoCredit"],
                                "usage_rights": info["allowCommercialUse"],
                                "base_model": sd_paths[info["modelVersions"][0]["baseModel"]],
                            }
                            if "format" in file_info["metadata"]:
                                updated_info[info["name"]][
                                    "lora_path"
                                ] = f'{file_info["downloadUrl"]}?type={file_info["type"]}'
                                f'&format={file_info["metadata"]["format"]}'
                            else:
                                updated_info[info["name"]][
                                    "lora_path"
                                ] = f'{file_info["downloadUrl"]}?type=SafeTensor'
                            if "trainedWords" in info["modelVersions"][0]:
                                updated_info[info["name"]]["trigger words"] = info["modelVersions"][
                                    0
                                ]["trainedWords"]
                        else:
                            if "lora_path" in self.implemeted_loras[model]:
                                updated_info[prep_name(model)] = self.implemeted_loras[model]
                                updated_info[prep_name(model)]["model_id"] = -1
                    except KeyError:
                        ret += "\n" + f"something did go wrong with model {model}!"
                        pass
            else:
                updated_info[prep_name(model)] = self.implemeted_loras[model]
        self.implemeted_loras = updated_info
        with open("lora_list.json", "w") as f:
            json.dump(self.implemeted_loras, f, indent=4)
        return ret

    def load_implemented_loras(self):
        global sd_paths
        with open("lora_list.json") as f:
            self.implemeted_loras = json.load(f)
        self._update_lora_info()
        self.prep_lora()

    def prep_lora(self):
        for lora in tqdm(self.implemeted_loras, "downloading loras"):
            if not os.path.exists(os.path.join("loras", lora)):
                os.mkdir(os.path.join("loras", lora))
                subprocess.run(
                    [
                        "wget",
                        "-O",
                        os.path.join("loras", lora, "lora.safetensors"),
                        self.implemeted_loras[lora]["lora_path"] + f"&token={config.CIVIT_KEY}",
                    ]
                )

    def _validate_model_config(self, config: dict) -> dict:
        if "torch_dtype" in config:
            config["torch_dtype"] = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }[config["torch_dtype"]]
        return config

    def _load_flux_pipeline(self):
        self.loaded_pipeline = "flux"
        pipe = FluxPipeline.from_pretrained(
            self.model_id, **self.config, safety_checker=None, token=config.HF_TOKEN
        )
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        self.pipe = pipe.to("cuda:1")
        torch.cuda.empty_cache()

    def _load_xl_pipeline(self):
        self.loaded_pipeline = "xl"
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
            del self.config["scheduler"]
        else:
            scheduler = "euler"
        if self.type == "prompt2img":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id, **self.config, token=config.HF_TOKEN
            )
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_id, **self.config, token=config.HF_TOKEN
            )

        pipe.scheduler = self.schedulers[scheduler].from_config(pipe.scheduler.config)
        pipe.unet.added_cond_kwargs = {"text_embeds": []}
        print("loaded model!")
        self.pipe = pipe.to("cuda")
        if (
            self.loaded_lora is not None
            and self.model_id == self.implemeted_loras[self.loaded_lora]["base_model"]
        ):
            self.pipe.load_lora_weights(
                os.path.join("..", "loras", self.loaded_lora, "lora.safetensors")
            )
        torch.cuda.empty_cache()

    def _load_long_pipeline(self):
        self.loaded_pipeline = "long"
        pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            **self.config,
            custom_pipeline="lpw_stable_diffusion",
            safety_checker=None,
        )
        self.pipe = pipe.to("cuda")
        if (
            self.loaded_lora is not None
            and self.model_id == self.implemeted_loras[self.loaded_lora]["base_model"]
        ):
            self.pipe.load_lora_weights(
                os.path.join("..", "loras", self.loaded_lora, "lora.safetensors")
            )
        torch.cuda.empty_cache()

    def _load_base_pipeline(self):
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
            del self.config["scheduler"]
        else:
            scheduler = "euler"
        self.loaded_pipeline = "base"
        if self.type == "prompt2img":
            pipe = StableDiffusionPipeline.from_pretrained(self.model_id, **self.config)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id, **self.config)
        pipe.scheduler = self.schedulers[scheduler].from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")
        torch.cuda.empty_cache()

    def set_model(self, config=None):
        reset = config["type"] != self.type
        if config is not None:
            if "model_id" in config and self.model_id != config["model_id"]:
                self.model_id = config["model_id"]
                reset = True
            if (
                "lora" in config
                and config["lora"] != self.loaded_lora
                and config["lora"] in self.implemeted_loras
            ):
                self.loaded_lora = config["lora"]
                reset = True
            if "lora" not in config and self.loaded_lora is not None:
                reset = True
                self.loaded_lora = None
            if "model_config" in config:
                model_config = self._validate_model_config(config["model_config"])
                if any([self.config[k] != model_config[k] for k in model_config]):
                    self.config = model_config
                    reset = True
        if reset:
            self.type = config["type"]
            if "xl" in self.model_id:
                self._load_xl_pipeline()
            elif "FLUX" in self.model_id:
                self._load_flux_pipeline()
            else:
                self._load_long_pipeline()
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

    def _pipeline_gen_call(self, args, config):
        if self.loaded_pipeline in ["long"]:
            if config["type"] == "img2img":
                args["image"] = args["image"][0]
                images = self.pipe.img2img(**args, max_embeddings_multiples=20).images
            elif config["type"] == "prompt2img":
                images = self.pipe.text2img(**args, max_embeddings_multiples=20).images
        else:
            images = self.pipe(**args).images
        return images

    def gen_image(self, prompt, config):
        # Reset timer on each image generation
        self.reset_timer()

        n = config["count_returned"]
        negative_prompt = config["prompt_config"]["negative_prompt"]
        del config["prompt_config"]["negative_prompt"]
        seed = config["seed"]
        if seed < 0:
            seed = np.random.randint(2**8)
        args = {
            "prompt": [prompt] * n,
            "negative_prompt": [negative_prompt] * n,
            "generator": [torch.Generator("cuda").manual_seed(i) for i in range(seed, seed + n)],
        } | config["prompt_config"]
        if "image" not in config:
            config["type"] = "prompt2img"
            args["width"] = int(config["width"] // 8 * 8)
            args["height"] = int(config["height"] // 8 * 8)
        else:
            config["type"] = "img2img"
            init_image = Image.open(BytesIO(config["image"])).convert("RGB")
            init_image.resize((config["width"], config["height"]))
            args["image"] = [init_image] * n
        self.set_model(config)
        if self.loaded_pipeline == "flux":
            del args["negative_prompt"]
        return self._pipeline_gen_call(args, config)


def image_grid(imgs):
    rows = int(np.ceil(len(imgs) / 4))
    cols = int(np.ceil(len(imgs) / rows))
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


model = DiffusionModel()
model_config = ["torch_dtype"]
prompt_config = ["num_inference_steps", "guidance_scale", "negative_prompt"]

app = FastAPI()
router = APIRouter()


class Item(BaseModel):
    model_id: str | None = "runwayml/stable-diffusion-v1-5"
    prompt: str | None = "Ein schöner Strand im Süden Schleswig-Holsteins, digitales Kunstwerk"
    torch_dtype: str | None = "float16"
    num_inference_steps: int | None = 20
    count_returned: int | None = 1
    seed: int | None = 0
    guidance_scale: float | None = 7.5
    negative_prompt: str | None = "blurry, low resolution, low quality"
    width: int | None = 512
    height: int | None = 512
    lora: str | None = None


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


@router.get("/llm_prompt_assistance")
async def llm_prompt_assistance(text: str, api_key: str = Depends(verify_api_key)):
    """Forward LLM prompt assistance request (legacy endpoint)."""
    answer = requests.get(f"{config.LLM_MIXTRAL_URL}/llm_prompt_assistance?text={text}")
    if answer.ok:
        return answer.content
    else:
        return "could not reach LLM"


@router.get("/llm_interface")
async def llm_interface(text: str, role: str = None, api_key: str = Depends(verify_api_key)):
    """Forward LLM interface request (legacy endpoint)."""
    answer = requests.get(f"{config.LLM_MIXTRAL_URL}/llm_interface?text={text}")
    if answer.ok:
        return answer.content
    else:
        return "could not reach LLM"


@router.post(
    "/post_config", responses={200: {"content": {"image/png": {}}}}, response_class=Response
)
async def get_image(
    item: Item = Depends(), image: UploadFile = File(None), api_key: str = Depends(verify_api_key)
):
    """Generate images using Stable Diffusion with optional LORA."""
    cfg = _refactor_config(vars(item))
    prompt = item.prompt
    del cfg["prompt"]
    if image is not None:
        cfg["image"] = await image.read()
    image = image_grid(model.gen_image(prompt, cfg))
    image = get_bytes_value(image)
    return Response(content=image, media_type="image/png")


@router.get("/get_model_embedding")
async def get_embedded_text(prompt: str, api_key: str = Depends(verify_api_key)):
    """Get text embeddings from current model."""
    embedding, embedded_text = model._prompt_to_embedding(prompt)
    return embedded_text


@router.get("/get_implemented_parameters")
async def get_parameters(api_key: str = Depends(verify_api_key)):
    """Get list of configurable parameters."""
    item = Item()
    parameters = _refactor_config(vars(item))
    return parameters


@router.get("/get_available_loras")
async def get_available_loras(api_key: str = Depends(verify_api_key)):
    """Get list of available LORA models."""
    return model.implemeted_loras


@router.get("/get_available_stable_diffs")
async def get_available_SDs(api_key: str = Depends(verify_api_key)):
    """Get list of available Stable Diffusion models."""
    global sd_paths
    return sd_paths


@router.post("/add_new_lora")
async def add_new_lora(name: str, api_key: str = Depends(verify_admin_key)):
    """Add a new LORA model from Civitai (admin only)."""
    answer = model.add_new_lora(name)
    return answer


@router.get("/buffer_status/")
async def get_buffer_status(api_key: str = Depends(verify_api_key)):
    """Get current buffer status for debugging."""
    return model.get_status()


app.include_router(router)
