import json
import os
import random
import torch 
from PIL import Image
from diffusers import StableDiffusionPipeline
from natsort import natsorted
from glob import glob
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display

def grid_img(imgs,rows=1,cols=3,scale=1):
    assert len(imgs) == rows * cols
    w,h = imgs[0].size
    w,h = int(w*scale), int(h*scale)

    grid = Image.new("RGB",size=(cols*w,rows*h))

    for i,img in enumerate(imgs):
        img = img.resize((w,h), Image.ADAPTIVE)
        grid.paste(img,box=(i % cols * w, i // cols * h))
    return grid

model_sd = "runwayml/stable-diffusion-v1-5"
output_dir = "AI_images\\training_images"

concept_list = [
    {
        "instance_prompt":"kratos",
        "class_prompt":"photo of a person",
        "instance_data_dir":"AI_images\\data\\training",
        "class_data_dir":"AI_images\\data\\person"
    }
]

for c in concept_list:
    os.makedirs(c["instance_data_dir"],exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concept_list, f,indent=4)

num_images = 4
num_class_images = num_images * 12
max_num_steps = num_images * 80
learning_rate = 1e-6 # 0.0000001
lrn_warmup_steps = int(max_num_steps/10)
print(num_images,num_class_images,max_num_steps,learning_rate,lrn_warmup_steps)

weights_dir = natsorted(glob(output_dir + os.sep + "*"))[-1]
weights_folder = output_dir
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key= lambda x:int(x))

imgs_test = []

for imgs,folder in enumerate(folders):
    folder_path = os.path.join(weights_folder,folder)
    image_folder= os.path.join(folder_path,"samples")
    images = [f for f in os.listdir(image_folder)]

    for i in images:
        img_path = os.path.join(image_folder,i)
        r = Image.open(img_path)
        imgs_test.append(r)

grid_img(imgs_test,rows=1,cols=4,scale=1)


prompt = "face portrait of kratos in the snow, realistic, hd, vivid, sunset"
negative_prompt = "bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet, blurry, low quality, low definition, lowres, out of frame, out of image, cropped, cut off, signature, watermark"
num_samples = 5
guidance_scale = 7.5
num_inference_steps = 30
height = 512
width = 512

seed = random.randint(0, 2147483647)
print("Seed: {}".format(str(seed)))
generator = torch.Generator(device='cuda').manual_seed(seed)
model_path = weights_dir
print(model_path)
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')

with autocast("cuda"), torch.inference_mode():
    imgs = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height, width=width,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images

for img in imgs:
    display(img)

prompt = ["photo of kratos person, closeup, mountain fuji in the background, natural lighting",
          "photo of kratos person in the desert, closeup, pyramids in the background, natural lighting, frontal face",
          "photo of kratos person in the forest, natural lighting, frontal face",
          "photo of kratos person as an astronaut, natural lighting, frontal face, closeup, starry sky in the background",
          "face portrait of kratos in the snow, realistic, hd, vivid, sunset"]

negative_prompt = ["bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet, blurry, low quality, low definition, lowres, out of frame, out of image, cropped, cut off, signature, watermark" ] * len(prompt)
num_samples = 1
guidance_scale = 7.5
num_inference_steps = 30
height = 512
width = 512

seed = random.randint(0, 2147483647)
print("Seed: {}".format(str(seed)))
generator = torch.Generator(device='cuda').manual_seed(seed)

with autocast("cuda"), torch.inference_mode():
    imgs = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height, width=width,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images

for img in imgs:
    display(img)