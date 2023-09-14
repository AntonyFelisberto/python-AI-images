from diffusers import StableDiffusionControlNetPipeline,ControlNetModel
import torch
import cv2
from PIL import Image
import numpy as np
from diffusers import UniPCMultistepScheduler

def grid_img(imgs, rows=1, cols=3, scale=1):
  assert len(imgs) == rows * cols

  w, h = imgs[0].size
  w, h = int(w*scale), int(h*scale)
  
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size
  
  for i, img in enumerate(imgs):
      img = img.resize((w,h), Image.LANCZOS)
      grid.paste(img, box=(i%cols*w, i//cols*h))
  return grid

def canny_edge(img,low_threshold=100,high_threshold=200):
   img = np.array(img)
   img = cv2.Canny(img,low_threshold,high_threshold)
   img = img[:,:,None]
   img = np.concatenate([img,img,img],axis=2)
   canny_img = Image.fromarray(img)
   return canny_img

controlnet_canny_model = 'lllyasviel/sd-controlnet-canny'
control_net_canny = ControlNetModel.from_pretrained(controlnet_canny_model, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                         controlnet=control_net_canny,
                                                         torch_dtype=torch.float16)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

img = Image.open("AI_images\\images\\dog.png")
canny_img = canny_edge(img)
canny_img.save("testes.jpg")
canny_img = canny_edge(img,200,455)
canny_img.save("testes.jpg")


device = "cuda"
pipe = pipe.to(device)
seed = 777

prompt = "realistic dog on the chair"
neg_prompt = ""
generator = torch.Generator(device=device).manual_seed(seed)
img = pipe(prompt=prompt, image=canny_img,negative_prompt=neg_prompt,generator=generator,num_inference_steps=20).images[0]
img.save("examples.png")


prompt = ["realistic dog on the chair","realistic dog on the chair while superheroes are fighting in the background"]
neg_prompt = ["blurred,lowres,bad anatomy,ugly, worst quality,monochrome,signature"] * len(prompt)
generator = torch.Generator(device=device).manual_seed(seed)
imgs = pipe(prompt=prompt, image=canny_img,negative_prompt=neg_prompt,generator=generator,num_inference_steps=20)
img = grid_img(imgs.images, 1 ,len(prompt),scale=0.75)
img.save("examples1.png")