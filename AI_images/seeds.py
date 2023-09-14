import torch 
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

pipe = pipe.to('cuda')

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

prompt = "trees"

seed = 777
generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompt,generator=generator).images[0]
type(img)
img.save("image.png")