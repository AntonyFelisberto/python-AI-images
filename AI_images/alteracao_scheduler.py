import torch 
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler,LMSDiscreteScheduler,EulerDiscreteScheduler,EulerAncestralDiscreteScheduler
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
#pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v2-1", torch_dtype=torch.float16) #versão 2
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler.compatibles

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

prompt = "photo of a futuristic city on another planet, realistic, full hd"
neg_prompt = 'buildings'
num_imgs = 3

imgs = pipe(prompt, negative_prompt = neg_prompt, num_images_per_prompt=num_imgs,num_inference_steps=30).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid.save("pcs.jpg")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
prompt = "photo of a futuristic city on another planet, realistic, full hd"
neg_prompt = 'buildings'
num_imgs = 3

imgs = pipe(prompt, negative_prompt = neg_prompt, num_images_per_prompt=num_imgs,num_inference_steps=30).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid.save("pcs21.jpg")

pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
prompt = "photo of a futuristic city on another planet, realistic, full hd"
neg_prompt = 'buildings'
num_imgs = 3

imgs = pipe(prompt, negative_prompt = neg_prompt, num_images_per_prompt=num_imgs,num_inference_steps=30).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid.save("pcs2.jpg")

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
prompt = "photo of a futuristic city on another planet, realistic, full hd"
neg_prompt = 'buildings'
num_imgs = 3

imgs = pipe(prompt, negative_prompt = neg_prompt, num_images_per_prompt=num_imgs,num_inference_steps=30).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid.save("pcs22.jpg")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
prompt = "photo of a futuristic city on another planet, realistic, full hd"
neg_prompt = 'buildings'
num_imgs = 3

imgs = pipe(prompt, negative_prompt = neg_prompt, num_images_per_prompt=num_imgs,num_inference_steps=30).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid.save("pcs23.jpg")