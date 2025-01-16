# Upscaler
# Closer to the original input image
import os
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)

from helpers.image_helper import (
    resize_images,
)

from helpers.upscaler_helper import (
    interrupt_callback,
)

# Models
controlnet_model = "models/controlnet_models/control_v11f1e_sd15_tile"
sd15_model = "models/sd15/photoRealV15_photorealv21"
vae_model = "models/vae/sd-vae-ft-mse"

# IP Adapter
ip_adapter_model = "models/ip_adapter/h94/IP_Adapter"
ip_adapter_subfolder = "models"
ip_adapter_weight_name = "ip-adapter_sd15.bin"

# noise scheduler
noise_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00095,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    steps_offset=1,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++"
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "models/ip_adapter/h94/IP_Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to("cuda")

# vae
vae = AutoencoderKL.from_pretrained(vae_model,
                                    torch_dtype=torch.float16).to("cuda")
# ControlNetModel
controlnet = ControlNetModel.from_pretrained(controlnet_model,
                                             torch_dtype=torch.float16).to("cuda")
# StableDiffusionControlNetImg2ImgPipeline
pipe_ctrl_img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    sd15_model,
    controlnet=controlnet,
    vae=vae,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    safety_checker=None,
    use_safetensors=True,
    torch_dtype=torch.float16
).to("cuda")
pipe_ctrl_img.enable_xformers_memory_efficient_attention()

# StableDiffusionImg2ImgPipeline
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    sd15_model,
    vae=vae,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    safety_checker=None,
    use_safetensors=True,
    torch_dtype=torch.float16
).to("cuda")
pipe_img2img.enable_xformers_memory_efficient_attention()

pipe_ctrl_img.load_ip_adapter(ip_adapter_model, subfolder=ip_adapter_subfolder, weight_name=ip_adapter_weight_name)

# Inputs
prompt = "best quality, high details"
negative_prompt = "poorly drawn, ugly, tiling, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy,  writing, calligraphy, sign, cut off"

pipe_ctrl_img.load_lora_weights("models/lora/more_details.safetensors")
pipe_ctrl_img.fuse_lora(lora_scale=0.90)

pipe_ctrl_img.set_ip_adapter_scale(0.80)
pipe_ctrl_img.enable_freeu(s1=0.90, s2=0.20, b1=1.50, b2=1.60)

# Input Image
input_image = Image.open("images/input_images/man-with-wine.jpg")

scale_factor = 1.5
resized_image = resize_images(input_image, scale_factor)

output_directory = "images/output_images"

# Image Generation

# Stage I - IMG2IMG
image_1 = pipe_img2img(
             prompt=prompt,
             negative_prompt=negative_prompt,
             image=resized_image,
             strength=0.40,
             num_inference_steps=50,
             guidance_scale=7.5,
             callback_on_step_end=interrupt_callback,
             generator=torch.manual_seed(0),
             output_type="pil"
             ).images[0]
torch.cuda.empty_cache()
image_1.save(os.path.join(output_directory, 'upscaled_1_img2img_1.jpg'), format="JPEG", quality=90)

scale_factor = 2
resized_image = resize_images(input_image, scale_factor)

# Stage II - CTRLIMG2IMG
image_ctrl_img_1 = pipe_ctrl_img(
             prompt=prompt,
             negative_prompt=negative_prompt,
             image=image_1,
             control_image=resized_image,
             ip_adapter_image=resized_image,
             width=resized_image.size[0],
             height=resized_image.size[1],
             strength=0.40,
             num_inference_steps=50,
             guidance_scale=4,
             controlnet_conditioning_scale=0.65,
             controlnet_guidance_start=0.0,
             controlnet_guidance_end=1.0,
             generator=torch.manual_seed(0),
             output_type="pil"
             ).images[0]
torch.cuda.empty_cache()
image_ctrl_img_1.save(os.path.join(output_directory, 'upscaled_1_ctrl_img_1.jpg'), format="JPEG", quality=90)

torch.cuda.empty_cache()
pipe_ctrl_img.unload_ip_adapter()
pipe_ctrl_img.unload_lora_weights()
pipe_ctrl_img.unfuse_lora()

del vae
del controlnet
del pipe_img2img
del pipe_ctrl_img
