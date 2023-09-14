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

prompt = "Lamborghini image with vin diesel in the car"
generator = torch.Generator(device=device).manual_seed(seed)
img = pipe(prompt=prompt, image=init_img,generator=generator,strength=0.85).images[0]
img.save("examples.png")

generator = torch.Generator(device=device).manual_seed(seed)
img = pipe(prompt=prompt, image=init_img,generator=generator,strength=0.75).images[0]
img.save("examples_2.png")

plt.figure(figsize=(18,8))
for i in range(1,6):
    strength_val = (i + 4) / 10
    generator = torch.Generator('cuda').manual_seed(seed)
    img = pipe(prompt,image=init_img,strength=strength_val,generator=generator).images[0]
    plt.subplot(1,5,i)
    plt.title("strenght: {}".format(strength_val))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    img.save(f"image{i}.png")