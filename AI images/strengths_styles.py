import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,safety_checker = None)
pipe = pipe.to(device)

seed = 777
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

init_img = Image.open("R.jfif")
init_img.thumbnail((512,512))

prompt = "watercolor painting Lamborghini image with vin diesel in the car"
generator = torch.Generator(device=device).manual_seed(seed)
img = pipe(prompt=prompt, image=init_img,generator=generator,strength=0.85).images[0]
img.save("examples.png")

prompt = "oil painting Lamborghini image with vin diesel in the car"
generator = torch.Generator(device=device).manual_seed(seed)
img = pipe(prompt=prompt, image=init_img,generator=generator,strength=0.8).images[0]
img.save("examples1.png")

prompt = "Lamborghini image with vin diesel in the car, van gog painting"
generator = torch.Generator(device=device).manual_seed(seed)
img = pipe(prompt=prompt, image=init_img,generator=generator,strength=1.0).images[0]
img.save("examples2.png")