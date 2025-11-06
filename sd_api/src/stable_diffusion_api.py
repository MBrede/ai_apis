"""
To start the API navigate to the scripts folder and call:
gunicorn stable_diffusion_api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:1234
this requires a gunicorn to be installed (pip install should do the trick)
"""

import torch
from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline,
                       StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler)
from diffusers import FluxPipeline
from fastapi import FastAPI, APIRouter, Response, UploadFile, File, Depends
from pydantic import BaseModel
from io import BytesIO
import numpy as np
from PIL import Image
import os
import subprocess
from tqdm import tqdm
import dotenv
import json
import requests
import re

dotenv.load_dotenv('.env')

sd_paths = {
    'SD 1.5': "runwayml/stable-diffusion-v1-5",
    'SD 2.0': "stabilityai/stable-diffusion-2",
    'SD 2.1': "stabilityai/stable-diffusion-2-1",
    'SDXL 1.0': "stabilityai/stable-diffusion-xl-base-1.0",
    'Flux.1 D': "black-forest-labs/FLUX.1-dev"
}


def prep_name(string):
    string = string.encode("ascii", "ignore").decode("ascii")
    string = re.sub(r"\s", '_', string.lower())
    return re.sub(r"[^a-z_0-9]", '', string.lower())


class DiffusionModel:

    schedulers = {
        'euler': EulerDiscreteScheduler
    }
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.loaded_lora = None
        self.config = {'torch_dtype': torch.float16}
        self.type = 'prompt2img'
        self.model_id = None
        self.set_model({'model_id': model_id,
                        'type': 'prompt2img'})
        self.load_implemented_loras()

    def add_new_lora(self, lora_name):
        search_progress = ''
        if lora_name not in self.implemeted_loras:
            self.implemeted_loras[lora_name] = {}
            if lora_name.isdigit():
                self.implemeted_loras[lora_name]['model_id'] = int(lora_name)
            search_progress = self._update_lora_info()
        return search_progress

    def _update_lora_info(self):
        updated_info = {}
        ret = ''
        for model in tqdm(self.implemeted_loras, 'updating info'):
            if 'base_model' not in self.implemeted_loras[model]:
                if 'model_id' in self.implemeted_loras[model]:
                    id = self.implemeted_loras[model]['model_id']
                    answer = requests.get(
                        f'https://civitai.com/api/v1/models/{id}'
                    )
                else:
                    answer = requests.get(
                        f'https://civitai.com/api/v1/models?query={model}&token={os.environ["civit_key"]}&nsfw=false')
                if answer.ok:
                    try:
                        info = json.loads(answer.content.decode('utf-8'))['items']
                        ret += '\n' + f'searchterm {model} lead to {len(info)} models'
                    except KeyError:
                        info = [json.loads(answer.content.decode('utf-8'))]
                    try:
                        if len(info) > 0:
                            info = info[0]
                            info['name'] = prep_name(info['name'])
                            ret += '\n' + f',collecting the first one: {info["name"]}'
                            file_info = info['modelVersions'][0]['files'][0]
                            updated_info[info['name']] = {
                                'model_id': info['id'],
                                'allow_no_mention': info['allowNoCredit'],
                                'usage_rights': info['allowCommercialUse'],
                                'base_model': sd_paths[info['modelVersions'][0]['baseModel']]
                            }
                            if "format" in file_info["metadata"]:
                                updated_info[info['name']]['lora_path']= f'{file_info["downloadUrl"]}?type={file_info["type"]}'
                                f'&format={file_info["metadata"]["format"]}'
                            else:
                                updated_info[info['name']]['lora_path'] = f'{file_info["downloadUrl"]}?type=SafeTensor'
                            if 'trainedWords' in info['modelVersions'][0]:
                                updated_info[info['name']]['trigger words']: info['modelVersions'][0]['trainedWords']
                        else:
                            if 'lora_path' in self.implemeted_loras[model]:
                                updated_info[prep_name(model)] = self.implemeted_loras[model]
                                updated_info[prep_name(model)]['model_id'] = -1
                    except KeyError:
                        ret += '\n' + f'something did go wrong with model {model}!'
                        pass
            else:
                updated_info[prep_name(model)] = self.implemeted_loras[model]
        self.implemeted_loras = updated_info
        with open('lora_list.json', 'w') as f:
            json.dump(self.implemeted_loras, f, indent=4)
        return ret

    def load_implemented_loras(self):
        global sd_paths
        with open('lora_list.json', 'r') as f:
            self.implemeted_loras = json.load(f)
        self._update_lora_info()
        self.prep_lora()

    def prep_lora(self):
        for lora in tqdm(self.implemeted_loras, 'downloading loras'):
            if not os.path.exists(os.path.join('loras', lora)):
                os.mkdir(os.path.join('loras', lora))
                subprocess.run(['wget', '-O', os.path.join('loras', lora, 'lora.safetensors'),
                                self.implemeted_loras[lora]['lora_path'] + f'&token={os.environ["civit_key"]}'])

    def _validate_model_config(self, config: dict) -> dict:
        if 'torch_dtype' in config:
            config['torch_dtype'] = {'float16': torch.float16,
                                     'float32': torch.float32,
                                     'float64': torch.float64}[config['torch_dtype']]
        return config

    def _load_flux_pipeline(self):
        self.loaded_pipeline = 'flux'
        pipe = FluxPipeline.from_pretrained(self.model_id, **self.config,
                                            safety_checker=None,
                                            token=os.environ["hf_token"])
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        self.pipe = pipe.to("cuda:1")
        torch.cuda.empty_cache()

    def _load_xl_pipeline(self):
        self.loaded_pipeline = 'xl'
        if 'scheduler' in self.config:
            scheduler = self.config['scheduler']
            del self.config['scheduler']
        else:
            scheduler = 'euler'
        if self.type == 'prompt2img':
            pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, **self.config,
                                                                    token=os.environ["hf_token"])
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.model_id, **self.config,
                                                                    token=os.environ["hf_token"])

        pipe.scheduler = self.schedulers[scheduler].from_config(pipe.scheduler.config)
        pipe.unet.added_cond_kwargs={'text_embeds': []}
        print('loaded model!')
        self.pipe = pipe.to("cuda")
        if self.loaded_lora is not None and self.model_id == self.implemeted_loras[self.loaded_lora]['base_model']:
            self.pipe.load_lora_weights(os.path.join('..', 'loras', self.loaded_lora, 'lora.safetensors'))
        torch.cuda.empty_cache()

    def _load_long_pipeline(self):
        self.loaded_pipeline = 'long'
        pipe = DiffusionPipeline.from_pretrained(self.model_id, **self.config,
                                                 custom_pipeline="lpw_stable_diffusion",
                                                 safety_checker=None)
        self.pipe = pipe.to("cuda")
        if self.loaded_lora is not None and self.model_id == self.implemeted_loras[self.loaded_lora]['base_model']:
            self.pipe.load_lora_weights(os.path.join('..', 'loras', self.loaded_lora, 'lora.safetensors'))
        torch.cuda.empty_cache()

    def _load_base_pipeline(self):
        if 'scheduler' in self.config:
            scheduler = self.config['scheduler']
            del self.config['scheduler']
        else:
            scheduler = 'euler'
        self.loaded_pipeline = 'base'
        if self.type == 'prompt2img':
            pipe = StableDiffusionPipeline.from_pretrained(self.model_id, **self.config)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id, **self.config)
        pipe.scheduler = self.schedulers[scheduler].from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")
        torch.cuda.empty_cache()

    def set_model(self, config=None):
        reset = config['type'] != self.type
        if config is not None:
            if 'model_id' in config and self.model_id != config['model_id']:
                self.model_id = config['model_id']
                reset = True
            if ('lora' in config and
                    config['lora'] != self.loaded_lora and
                    config['lora'] in self.implemeted_loras):
                self.loaded_lora = config['lora']
                reset = True
            if ('lora' not in config and
                    self.loaded_lora is not None):
                reset = True
                self.loaded_lora = None
            if 'model_config' in config:
                model_config = self._validate_model_config(config['model_config'])
                if any([self.config[k] != model_config[k] for k in model_config]):
                    self.config = model_config
                    reset = True
        if reset:
            self.type = config['type']
            if 'xl' in self.model_id:
                self._load_xl_pipeline()
            elif 'FLUX' in self.model_id:
                self._load_flux_pipeline()
            else:
                self._load_long_pipeline()
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False

    def _prompt_to_embedding(self, prompt):
        tokens = self.pipe.tokenizer(prompt, return_tensors="pt")
        prompt_embeds = self.pipe.text_encoder(
            tokens["input_ids"][:, 0:self.pipe.tokenizer.model_max_length].to("cuda"),
            attention_mask=tokens["attention_mask"][:, 0:self.pipe.tokenizer.model_max_length].to("cuda")
        )
        prompt_embeds = prompt_embeds[0]
        embedded_text = self.pipe.tokenizer.batch_decode(
            tokens["input_ids"][:, 1:(self.pipe.tokenizer.model_max_length - 1)]
        )
        return prompt_embeds.tolist(), embedded_text

    def _pipeline_gen_call(self, args, config):
        if self.loaded_pipeline in ['long']:
            if config['type'] == 'img2img':
                args['image'] = args['image'][0]
                images = self.pipe.img2img(**args,
                                           max_embeddings_multiples=20).images
            elif config['type'] == 'prompt2img':
                images = self.pipe.text2img(**args,
                                            max_embeddings_multiples=20).images
        else:
            images = self.pipe(**args).images
        return images

    def gen_image(self, prompt, config):
        n = config['count_returned']
        negative_prompt = config['prompt_config']['negative_prompt']
        del config['prompt_config']['negative_prompt']
        seed = config['seed']
        if seed < 0:
            seed = np.random.randint(2**8)
        args = {'prompt': [prompt] * n,
                'negative_prompt': [negative_prompt] * n,
                'generator': [torch.Generator("cuda").manual_seed(i)
                              for i in range(seed, seed + n)]} | config['prompt_config']
        if 'image' not in config:
            config['type'] = 'prompt2img'
            args['width'] = int(config['width'] // 8 * 8)
            args['height'] = int(config['height'] // 8 * 8)
        else:
            config['type'] = 'img2img'
            init_image = Image.open(BytesIO(config['image'])).convert("RGB")
            init_image.resize((config['width'], config['height']))
            args['image'] = [init_image] * n
        self.set_model(config)
        if self.loaded_pipeline == 'flux':
            del args['negative_prompt']
        return self._pipeline_gen_call(args, config)


def image_grid(imgs):
    rows = int(np.ceil(len(imgs) / 4))
    cols = int(np.ceil(len(imgs) / rows))
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


model = DiffusionModel()
model_config = ['torch_dtype']
prompt_config = ['num_inference_steps', 'guidance_scale', 'negative_prompt']

app = FastAPI()
router = APIRouter()


class Item(BaseModel):
    model_id: str | None = "runwayml/stable-diffusion-v1-5"
    prompt: str | None = 'Ein schöner Strand im Süden Schleswig-Holsteins, digitales Kunstwerk'
    torch_dtype: str | None = 'float16'
    num_inference_steps: int | None = 20
    count_returned: int | None = 1
    seed: int | None = 0
    guidance_scale: float | None = 7.5
    negative_prompt: str | None = 'blurry, low resolution, low quality'
    width: int | None = 512
    height: int | None = 512
    lora: str | None = None


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


@router.get("/llm_prompt_assistance")
async def llm_prompt_assistance(text: str):
    answer = requests.get('http://149.222.209.66:8000/llm_prompt_assistance?text={}'.format(text))
    if answer.ok:
        return answer.content
    else:
        return "could not reach LLM"


@router.get("/llm_interface")
async def llm_interface(text: str, role: str = None):
    answer = requests.get('http://149.222.209.66:8000/llm_interface?text={text}'
                          .format(text=text))
    if answer.ok:
        return answer.content
    else:
        return "could not reach LLM"


@router.post("/post_config",
             responses={
                 200: {
                     "content": {"image/png": {}}
                 }
             },
             response_class=Response
             )
async def get_image(item: Item = Depends(), image: UploadFile = File(None)):
    config = _refactor_config(vars(item))
    prompt = item.prompt
    del config['prompt']
    if image is not None:
        config['image'] = await image.read()
    image = image_grid(model.gen_image(prompt, config))
    image = get_bytes_value(image)
    return Response(content=image, media_type='image/png')


@router.get("/get_model_embedding")
async def get_embedded_text(prompt: str):
    embedding, embedded_text = model._prompt_to_embedding(prompt)
    return embedded_text


@router.get("/get_implemented_parameters")
async def get_parameters():
    item = Item()
    parameters = _refactor_config(vars(item))
    return parameters


@router.get("/get_available_loras")
async def get_available_loras():
    return model.implemeted_loras


@router.get("/get_available_stable_diffs")
async def get_available_SDs():
    global sd_paths
    return sd_paths


@router.get("/add_new_lora",
            include_in_schema=False)
async def add_new_lora(name):
    answer = model.add_new_lora(name)
    return answer


app.include_router(router)
