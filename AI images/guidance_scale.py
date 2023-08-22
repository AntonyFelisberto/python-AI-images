import torch 
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,safety_checker = None) #safety_checker serve para que o gerador possa gerar coisas não consideradas em ambientes de trabalho

pipe = pipe.to('cuda')

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

prompt = "men asking women to marry him"
seed = 777
plt.figure(figsize=(18,8))
for i in range(1,6):
    n_guidance = i + 4
    generator = torch.Generator('cuda').manual_seed(seed)
    img = pipe(prompt,guidance_scale=n_guidance,generator=generator).images[0]
    plt.subplot(1,5,i)
    plt.title("num_inference_steps: {}".format(n_guidance))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    img.save(f"image{i}.png")