import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device)

seed = 777
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

init_img = Image.open("IMG_20190916_003427.jpg")
init_img.thumbnail((512,512))
init_img.save("examples.png")