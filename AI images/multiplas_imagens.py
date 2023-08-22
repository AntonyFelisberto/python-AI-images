
from PIL import Image
import torch 
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

def grid_img(imgs,rows=1,cols=3,scale=1):
    assert len(imgs) == rows * cols
    w,h = imgs[0].size
    w,h = int(w*scale), int(h*scale)

    grid = Image.new("RGB",size=(cols*w,rows*h))

    for i,img in enumerate(imgs):
        img = img.resize((w,h), Image.ADAPTIVE)
        grid.paste(img,box=(i % cols * w, i // cols * h))
    return grid

num_imgs = 3
prompt = 'HORSE AGAINST SERPENT'
imgs = pipe(prompt,num_images_per_prompt=num_imgs).images
grid = grid_img(imgs,rows=1,cols=3,scale=0.75)
grid.save("image.png")