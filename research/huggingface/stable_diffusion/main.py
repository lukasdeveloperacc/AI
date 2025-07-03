from diffusers import DiffusionPipeline, ImagePipelineOutput

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id)
prompt = "portrait photo of a old warrior chief"
pipeline = pipeline.to("cuda")

import torch

generator = torch.Generator("cuda").manual_seed(0)
result: ImagePipelineOutput = pipeline(prompt, generator=generator)
images = result.images
image = images[0]
print(type(image))
image.save("test.png")
