import requests
import os
from PIL import Image
from io import BytesIO
from numpy import random
from tqdm import tqdm
import json
def api_request(image_path: str = None,
                model_id: str | None = 'stabilityai/stable-diffusion-2-1',
                prompt: str | None = 'Ein schöner Strand im Süden Schleswig-Holsteins, digitales Kunstwerk',
                torch_dtype: str | None = 'float16',
                num_inference_steps: int | None = 20,
                count_returned: int | None = 4,
                seed: int | None = 0,
                guidance_scale: float | None = 7.5,
                negative_prompt: str | None = 'blurry, low resolution, low quality',
                width: int | None = 512,
                height: int | None = 512):
    args = locals()
    if args['image_path'] is not None:
        image_path = args['image_path']
        name_img = os.path.basename(image_path)
        files = {'image': (name_img, open(image_path, 'rb'), 'multipart/form-data', {'Expires': '0'})}
        del args['image_path']
    else:
        files = {}
    url = f"http://149.222.209.100:8000/post_config?{'&'.join([f'{k}={args[k]}' for k in args])}"
    response = requests.post(url,  files=files)
    if not response.ok:
        raise ValueError("API did return an error! Check your parameters!")
    return Image.open(BytesIO(response.content))


def data_request():
    texts = ["Generate a series of concise, creative descriptions that can be used as prompts for generating images. Each description should be short, ideally a single sentence, and vividly capture a scene, character, or concept. Not all descriptions need to specify an art style; focus instead on creating a mix of descriptions—some that suggest a style and others that just set the scene or evoke an emotion. Ensure each description is unique and varied, providing a broad range of inspiration for visual artworks.",
             "Generate a series of creative and detailed descriptions for images that can be used to train an AI model in generating artwork. Each description should vividly depict a scene, character, or abstract idea that can be visually represented. Focus on including varied settings, moods, objects, and interactions to ensure a diverse dataset. Use clear and engaging language that can easily be translated into visual elements by an artist or a generative model. Aim for a balanced mix of realism and imagination in each description to inspire unique and compelling artworks.",
             "Generate a series of short descriptions of places, people and scenes.",
             "Generate a series of long, creative descriptions of places, people and scenes.",
             "Generate a series of descriptions of places, people and scenes, each no longer than two sentences."]
    sample = texts[random.randint(0,len(texts))]
    instruction = "{} Seperate the examples by an ';'.".format(sample)
    url = f"http://149.222.209.100:1234/llm_interface?temp=0.5&text={instruction}"
    response = requests.get(url)
    if not response.ok:
        raise ValueError("API did return an error! Check your parameters!")
    return response.text.split(';')


if __name__ == '__main__':
    # image = api_request(image_path='johannes.jpg')
    texts = []
    for i in tqdm(range(1000), desc='Generating texts...'):
        try:
            texts += data_request()
        except ValueError:
            pass
    prompts = []
    for text in tqdm(texts, desc="Generating prompts..."):
        response = requests.get(f"http://149.222.209.100:1234/llm_prompt_assistance", params={"text": text})
        if response.ok:
            prompts.append({'text': text,
                            'prompt': response.text.split('Description:')[-1][:-1]})
    with open('data/prompts.json', 'w') as f:
        json.dump(prompts, f)
