import torch 
from PIL import Image
from diffusers import StableDiffusionPipeline,EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler

from diffusers import DPMSolverMultistepScheduler

def grid_img(imgs,rows=1,cols=3,scale=1):
    assert len(imgs) == rows * cols
    w,h = imgs[0].size
    w,h = int(w*scale), int(h*scale)

    grid = Image.new("RGB",size=(cols*w,rows*h))

    for i,img in enumerate(imgs):
        img = img.resize((w,h), Image.ADAPTIVE)
        grid.paste(img,box=(i % cols * w, i // cols * h))
    return grid

seed = 777
anything = StableDiffusionPipeline.from_pretrained("cag/anything-v3-1", torch_dtype=torch.float16)  
anything = anything.to("cuda")
anything.enable_attention_slicing()
anything.scheduler = DPMSolverMultistepScheduler.from_config(anything.scheduler.config)

prompt = ["darth vader in the desert, medium shot, cinematic, red tones, masterpiece, best quality, illustration, beautiful detailed, finely detailed, dramatic light, intricate details",
          "space station orbiting earth, medium shot, colorful, masterpiece, best quality, illustration, beautiful detailed, finely detailed, dramatic light, intricate details",
          "female wizard wearing a red coat, medium shot, masterpiece, best quality, illustration, beautiful detailed, finely detailed, dramatic light, intricate details",
          "an orange cat reading a book in space"]
neg_prompt = ["bad drawing, watermark, poorly drawn hands"] * len(prompt)
generator = torch.Generator("cuda").manual_seed(seed)
anything.safety_checker = None
imgs = anything(prompt, 
          negative_prompt=neg_prompt, 
          generator=generator,
          guidance_scale=12,
          num_inference_steps=50).images 
  
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("imagens.png")

protogen = StableDiffusionPipeline.from_pretrained("darkstorm2150/Protogen_x3.4_Official_Release", torch_dtype=torch.float16)  
protogen = protogen.to("cuda")
protogen.enable_attention_slicing()
protogen.scheduler = DPMSolverMultistepScheduler.from_config(protogen.scheduler.config)

prompt = ["orange cat reading a book in space",
          "darth vader in ancient egypt, medium shot",
          "photo of a delorean in the canyons"]
neg_prompt = ["bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet"] * len(prompt)

generator = torch.Generator("cuda").manual_seed(seed)
protogen.safety_checker = None
imgs = protogen(prompt, 
          negative_prompt=neg_prompt, 
          generator=generator,
          num_inference_steps=25).images 
  
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("pros1.png")

ad = StableDiffusionPipeline.from_pretrained("wavymulder/Analog-Diffusion", torch_dtype=torch.float16)  
ad = ad.to("cuda")
ad.enable_attention_slicing()
ad.scheduler = EulerAncestralDiscreteScheduler.from_config(ad.scheduler.config)

prompt = ["analog style orange cat reading a book in space",
          "analog style cozy comfy cabin interior",
          "analog style portrait of a girl, purple and blue lights"]

neg_prompt = [""] * 3

generator = torch.Generator("cuda").manual_seed(seed)
ad.safety_checker = None
imgs = ad(prompt, 
          negative_prompt=neg_prompt, 
          generator=generator,
          num_inference_steps=35).images 
  
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("pros2.png")

ds = StableDiffusionPipeline.from_pretrained("Lykon/DreamShaper", torch_dtype=torch.float16)  
ds = ds.to("cuda")
ds.enable_attention_slicing()
ds.scheduler = EulerAncestralDiscreteScheduler.from_config(ds.scheduler.config)

prompt = ["analog style orange cat reading a book in space",
          "darth vader in ancient egypt, medium shot",
          "photo of a delorean in the canyons"]

neg_prompt = [""] * 3

generator = torch.Generator("cuda").manual_seed(seed)
ds.safety_checker = None
imgs = ds(prompt, 
          negative_prompt=neg_prompt, 
          generator=generator,
          num_inference_steps=35).images 
  
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("pros3.png")

ds = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V1.4", torch_dtype=torch.float16)  
ds = ds.to("cuda")
ds.enable_attention_slicing()
ds.scheduler = EulerAncestralDiscreteScheduler.from_config(ds.scheduler.config)

prompt = ["orange cat reading a book in space, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
          "darth vader in ancient egypt, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
          "delorean in the canyons, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"]

neg_prompt = [""] * 3

generator = torch.Generator("cuda").manual_seed(seed)
ds.safety_checker = None
imgs = ds(prompt, 
          negative_prompt=neg_prompt, 
          generator=generator,
          num_inference_steps=35).images 
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("pros4.png")