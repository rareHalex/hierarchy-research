import random
import os
import requests
from pathlib import Path


def get_prompts():
    if not os.path.exists("LOC_synset_mapping.txt"):
        r = requests.get("https://raw.githubusercontent.com/formigone/tf-imagenet/master/LOC_synset_mapping.txt")

        with open("LOC_synset_mapping.txt", "wb") as f:
            f.write(r.content)

    synsets = {
        k: v
        for k, v in [
            o.split(",")[0].split(" ", maxsplit=1)
            for o in Path("LOC_synset_mapping.txt").read_text().splitlines()
        ]
    }
    imagenet_classes = list(synsets.values())
    return imagenet_classes


imagenet_classes = get_prompts()


def imagenet_animal_prompts():
    animal = random.choice(imagenet_classes[:397])
    prompts = f'{animal}'
    return prompts


def decoding_fn(latents,pipe):
    images = pipe.vae.decode(1 / 0.18215 * latents.cuda()).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images