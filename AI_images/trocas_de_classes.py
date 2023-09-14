import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

seed = 888
inpainting_model = "runwayml/stable-diffusion-inpainting"
device = "cuda"
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
generator = torch.Generator(device=device).manual_seed(seed)

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

img_path = "dog.png"
img = Image.open(img_path)

mask_path = "dogs_refor.png"
img_mask = Image.open(mask_path)

w,h = img.size[0],img.size[1]

prompt = "nothing"

result_img_one = pipe(prompt=prompt,image=img,mask_image=img_mask,width=w,height=h,generator=generator).images[0]
result_img_one.save("test.png")

prompt = "an eletric guitar"
result_img_two = pipe(prompt=prompt,image=img,mask_image=img_mask,width=w,height=h,generator=generator).images[0]
result_img_two.save("test_2.png")

prompt = "a transformer"
result_img_three = pipe(prompt=prompt,image=img,mask_image=img_mask,width=w,height=h,generator=generator).images[0]
result_img_three.save("test_3.png")

imgs = [img,result_img_one,result_img_two,result_img_three]
grid_img(imgs,rows=1,cols=len(imgs),scale=0.75).save("archives.png")


prompt = "a giant spider"
num_imgs = 3
result_img_four = pipe(
    prompt=prompt,
    image=img,
    mask_image=img_mask,
    width=w,
    height=h,
    num_images_per_prompt=num_imgs,
    generator=generator).images 

multiples = grid_img(result_img_four, rows=1, cols=len(result_img_four), scale=0.75)
multiples.save("resources.png")