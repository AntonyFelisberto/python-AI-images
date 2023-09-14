from diffusers import StableDiffusionControlNetPipeline,ControlNetModel
import torch
import cv2
from PIL import Image
import numpy as np
from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from diffusers import DEISMultistepScheduler,EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

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

pose_model = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
controlnet_pose_model = ControlNetModel.from_pretrained('thibaud/controlnet-sd21-openpose-diffusers', torch_dtype=torch.float16)
sd_controlpose = StableDiffusionControlNetPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base',
                                                                   controlnet=controlnet_pose_model,
                                                                   torch_dtype=torch.float16)
sd_controlpose.enable_model_cpu_offload()
sd_controlpose.enable_attention_slicing()
sd_controlpose.enable_xformers_memory_efficient_attention()
sd_controlpose.scheduler = DEISMultistepScheduler.from_config(sd_controlpose.scheduler.config)

img_pose = Image.open("AI_images\images\OIPS.jfif")
pose = pose_model(img_pose)
img = grid_img([img_pose,pose], 1 ,2 ,scale=0.75)
img.save("poses.png")

device = "cuda"
pipe = pipe.to(device)
seed = 777

prompt = "realistic woman dancing"
neg_prompt = ""
generator = torch.Generator(device=device).manual_seed(seed)
imgs = sd_controlpose(prompt=prompt, image=pose,negative_prompt=neg_prompt,num_images_per_prompt=4,generator=generator,num_inference_steps=20)
img = grid_img(imgs.images, 1 ,4 ,scale=0.75)
img.save("examples.png")
 
urls = "yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"
imgs = [
    load_image("https://hf.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url) for url in urls
]
poses = [pose_model(img) for img in imgs]
img = grid_img(poses, 1 ,4 ,scale=0.75)
img.save("examples2.png")
