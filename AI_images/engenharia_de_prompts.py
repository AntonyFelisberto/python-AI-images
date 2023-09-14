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

prompt = "an orange cat"
generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompt,generator=generator).images[0]
img.save(f"image.png")

prompt = ["an orange cat","an orange cat reading a book in the kitchen","an orange cat reading a book in space"]
h,w = 768,512
generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompt,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompt),scale=0.5)
grid.save(f"image_2.png")

prompt = ["photo of an orange cat reading a book","oil painting of an orange cat reading a book in the kitchen","drawing of an orange cat reading a book in space"]
h,w = 768,512
generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompt,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompt),scale=0.5)
grid.save(f"image_3.png")

base = "orange cat reading a book in space"
m_ = [", Oil painting",
      ", digital painting",
      ", underwater steampunk"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompt),scale=0.5)
grid.save(f"image_4.png")

base = "orange cat reading a book in space"
m_ = [", Modernist",
      ", impressionist",
      ", realistic"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompt),scale=0.5)
grid.save(f"image_4.png")

base = "orange cat reading a book in space, realistic"
m_ = [", purple",
      ", red colors",
      ", black and white"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompt),scale=0.5)
grid.save(f"image_5.png")

base = "orange cat reading a book in space, realistic"
m_ = [", by Van Gogh",
      ", by Sandro Botticelli",
      ", by Monet",
      ", by Da Vinci"]

prompts = []
for m in m_:
    prompts.append(base+m)

generator = torch.Generator('cuda').manual_seed(seed)
img = pipe(prompts,generator=generator).images
grid = grid_img(img,rows=1,cols=len(prompt),scale=0.5)
grid.save(f"image_6.png")