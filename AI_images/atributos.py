import torch 
from PIL import Image
from diffusers import StableDiffusionPipeline

seed = 777

def grid_img(imgs,rows=1,cols=3,scale=1):
    assert len(imgs) == rows * cols
    w,h = imgs[0].size
    w,h = int(w*scale), int(h*scale)

    grid = Image.new("RGB",size=(cols*w,rows*h))

    for i,img in enumerate(imgs):
        img = img.resize((w,h), Image.ADAPTIVE)
        grid.paste(img,box=(i % cols * w, i // cols * h))
    return grid

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to('cuda')

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

base = "a futuristic city on another planet at dawn"
m_ = [", Oil painting",
      ", digital painting",
      ", underwater steampunk"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompts),scale=0.5)
grid.save(f"image_4.png")

base = "a futuristic city on another planet at dawn"
m_ = [", unreal engine",
      ", sharp focus",
      ", vray"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompts),scale=0.5)
grid.save(f"image_4.png")

base = "a futuristic city on another planet at dawn"
m_ = [", unsplash, stunningly beautiful, award winning photom tilt-shift",
      ", dramatic",
      ", low angle shot"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompts),scale=0.5)
grid.save(f"image_5.png")
