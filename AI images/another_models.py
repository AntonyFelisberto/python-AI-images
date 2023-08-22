import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

modi = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/mo-di-diffusion", torch_dtype=torch.float16)
modi = modi.to("cuda")

seed = 777
device = "cuda"

init_img = Image.open("R.jfif")

prompt = "modern disney, Lamborghini image with vin diesel in the car"
generator = torch.Generator(device=device).manual_seed(seed)
img = modi(prompt=prompt, image=init_img,generator=generator,strength=0.7,guidance_scale=7.5).images[0]
img.save("examples1.png")

init_img = Image.open("OIP.jfif")

prompt = "modern disney, boy"
generator = torch.Generator(device=device).manual_seed(seed)
img = modi(prompt=prompt, image=init_img,generator=generator,strength=1.0,guidance_scale=7.5).images[0]
img.save("examples2.png")

gb = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-diffusion", torch_dtype=torch.float16)
gb = gb.to("cuda")
seed = 777

init_img = Image.open("OIP.jfif")

prompt = "ghibli style, boy"
generator = torch.Generator(device=device).manual_seed(seed)
img = gb(prompt=prompt, image=init_img,generator=generator,strength=1.0,guidance_scale=7.5).images[0]
img.save("examples3.png")

gb.scheduler =  EulerAncestralDiscreteScheduler.from_config(gb.scheduler.config)

prompt = "ghibli style, boy"
generator = torch.Generator(device=device).manual_seed(seed)
img = gb(prompt=prompt, image=init_img,generator=generator,strength=1.0,guidance_scale=7.5).images[0]
img.save("examples4.png")