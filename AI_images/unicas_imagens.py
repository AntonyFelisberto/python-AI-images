#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install diffusers
#pip install xformers
#pip install transformers
#pip install git+https://github.com/Keith-Hon/bitsandbytes-windows.git
import torch 
from diffusers import StableDiffusionPipeline

print(torch.cuda.is_available())

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

pipe = pipe.to('cuda')

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

prompt = "an apple"

img = pipe(prompt).images[0]
type(img)

prompt = 'photograph of an apple'
img = pipe(prompt).images[0]
img.save("apple.png")

prompt = 'Blue Trees'
img = pipe(prompt).images[0]
img.save("Trees.png")
print(img.size)