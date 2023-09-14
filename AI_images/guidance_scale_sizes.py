import torch 
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

pipe = pipe.to('cuda')

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

prompt = "MONTAINS"
seed = 777
h,w = 512,768
generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompt,height=h,width=w,generator=generator).images[0]
img.save(f"image.png")

h,w = 768,512
generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompt,height=h,width=w,generator=generator).images[0]
img.save(f"image_2.png")