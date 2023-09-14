import torch 
from PIL import Image
from diffusers import StableDiffusionPipeline,EulerAncestralDiscreteScheduler

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

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,safety_checker = None)
pipe = pipe.to('cuda')

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

prompt = "new york city"
num_imgs = 3
generator = torch.Generator('cuda').manual_seed(seed)
ng_prompt = "cars"
imgs = pipe(prompt,num_images_per_prompt=num_imgs,generator=generator,negative_prompt=ng_prompt).images
grid = grid_img(imgs,rows=1,cols=num_imgs,scale=0.5)
grid.save(f"image_5.png")

prompt = "a young woman, front face, wearing a blue dress, natural light"
neg_prompt = "bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet"
img = pipe(prompt,negative_prompt=neg_prompt, generator=generator).images[0]
img.save("womans.jpg")


sd21 = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)  
sd21.scheduler = EulerAncestralDiscreteScheduler.from_config(sd21.scheduler.config)
print(sd21.scheduler)
sd21 = sd21.to("cuda")
sd21.enable_attention_slicing()

prompt = ["an orange cat ", "an orange cat reading a book in the kitchen", "an orange cat reading a book in space"]
neg_prompt = ["bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet"] * len(prompt)
imgs = sd21(prompt, negative_prompt=neg_prompt,  generator=torch.Generator("cuda").manual_seed(seed),  num_inference_steps=35).images 
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.5)
grid.save("cats.jpg")

prompt = "photo of young woman sitting by window with headphones, realistic, golden hour, rim lighting"
neg_prompt = "bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet"
img = sd21(prompt, negative_prompt=neg_prompt, generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=35).images[0]
img.save("woman.jpg")

prompt = "a young man in the street, wearing a suit and sunglasses, side light, golden hour"
neg_prompt = "ugly, tiling, closed eyes, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
img = sd21(prompt, negative_prompt=neg_prompt, generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=35).images[0]
img.save("mens.jpg")

prompt = "oil painting walter white wearing a suit and black hat, by Alphonse Mucha, face portrait, in the desert, realistic, vivid, fantasy, Surrealist"
neg_prompt = "out of frame, out of image, unrealistic, bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet"
num_imgs = 3
generator = torch.Generator("cuda").manual_seed(seed)
imgs = sd21(prompt, num_images_per_prompt=num_imgs, generator=generator, negative_prompt = neg_prompt, num_inference_steps=35).images
grid = grid_img(imgs, rows=1, cols=num_imgs, scale=0.5)
grid.save("say_my_name.png")

prompt = "oil painting walter white wearing a suit and black hat, by Alphonse Mucha, face portrait, in the desert, realistic, vivid, fantasy, Surrealist"
neg_prompt = "out of frame, out of image, unrealistic, bad anatomy, ugly, deformed, desfigured, distorted face, poorly drawn hands, poorly drawn face, poorly drawn feet"
num_imgs = 3
generator = torch.Generator("cuda").manual_seed(seed)
imgs = sd21(prompt, num_images_per_prompt=num_imgs, generator=generator, negative_prompt = neg_prompt, num_inference_steps=50).images
grid = grid_img(imgs, rows=1, cols=num_imgs, scale=0.5)
grid.save("say_my_name2.png")

prompt = ["photo of young woman sitting by window with headphones, realistic, golden hour, rim lighting","a young female, highlights in hair, 35mm, medium shot, sitting outside restaurant, brown eyes, wearing a dress, side light"]
neg_prompt = ["ugly, tiling, closed eyes, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"] * len(prompt)
num_imgs = 3
generator = torch.Generator("cuda").manual_seed(seed)
imgs = sd21(prompt, num_images_per_prompt=num_imgs, generator=generator, negative_prompt = neg_prompt, num_inference_steps=35).images
grid = grid_img(imgs, rows=len(prompt), cols=num_imgs, scale=0.5)
grid.save("womans.png")

prompt = ["photograph of snow in paris, realistic, landscape, sunset",
          "photograph of a mountain landscape during sunset, above the clouds"]
num_imgs = 2
h, w = 512, 768
generator = torch.Generator("cuda").manual_seed(seed)
imgs = pipe(prompt, num_images_per_prompt=num_imgs, generator=generator, height=h, width=w, negative_prompt = neg_prompt, num_inference_steps=35).images
grid = grid_img(imgs, rows=len(prompt), cols=num_imgs, scale=0.75)
grid.save("paris.png")

prompt = ["3d rendering of an apple, blender, realistic, highly detailed",
          "old blue chevette, 3d rendering, blender, realistic, highly detailed, 8k, vray"]
neg_prompt = ["ugly, overexposed, not centered, low resolution"] * len(prompt)
generator = torch.Generator("cuda").manual_seed(seed)
imgs = pipe(prompt, generator=generator, negative_prompt=neg_prompt, num_inference_steps=50).images
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("applications.png")

prompt = ["sticker design, an orange cat wearing sunglasses, cute vector, 8K, graphic design, beautiful design",
          "cactus sticker, cute vector, 8K, graphic design"
          ]
neg_prompt = ["bad design, bad drawing, watermark"] * len(prompt) 
generator = torch.Generator("cuda").manual_seed(seed)
imgs = sd21(prompt, generator=generator, negative_prompt = neg_prompt, num_inference_steps=35).images
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.5)
grid.save("arts.png")

prompt = ["a highly detailed epic cinematic concept art an alien pyramid landscape, art station, landscape, concept art, illustration, highly detailed artwork cinematic, hyper realistic painting", 
          "futuristic rio de janeiro, realistic, art station, cinematic, concept art, skyscrapers, landscape, highly detailed artwork"
          ]
generator = torch.Generator("cuda").manual_seed(seed)
imgs = pipe(prompt, generator=generator, num_inference_steps=50).images
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.75)
grid.save("countries.png")

prompt = ["a professional photograph of an italian architectural modern home, soft render, sunset, golden hour",
          "a professional photograph of an modern house, red door, soft render, golden hour"]
neg_prompt = ["watermark"] * len(prompt)
generator = torch.Generator("cuda").manual_seed(seed)
imgs = sd21(prompt, generator=generator,  negative_prompt = neg_prompt, num_inference_steps=35).images
grid = grid_img(imgs, rows=1, cols=len(prompt), scale=0.5)
grid.save("architetures.png")