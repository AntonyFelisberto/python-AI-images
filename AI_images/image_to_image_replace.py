import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

instruct = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16)
instruct.to("cuda")
instruct.scheduler = EulerAncestralDiscreteScheduler.from_config(instruct.scheduler.config)

init_img = Image.open("R.jfif")
init_img.thumbnail((512,512))

seed = 777
device = "cuda"
prompt = "put vin diesel in the car"
generator = torch.Generator(device=device).manual_seed (seed)
result_img = instruct(prompt, image=init_img,num_inference_steps=20,image_guidance_scale=1,generator=generator).images[0]
result_img.save("example.jpg")

prompt = "add fireworks in the sky"
generator = torch.Generator(device=device).manual_seed (seed)
result_img = instruct(prompt, image=init_img,num_inference_steps=20,image_guidance_scale=1,generator=generator).images[0]
result_img.save("example2.jpg")

prompt = "add spaceships in the sky"
generator = torch.Generator(device=device).manual_seed (seed)
result_img = instruct(prompt, image=init_img,num_inference_steps=15,image_guidance_scale=1,generator=generator).images[0]
result_img.save("example2.jpg")