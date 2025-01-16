'''
############ Tensor_upscaler ############

- https://huggingface.co/docs/diffusers/v0.26.3/en/api/models/controlnet#diffusers.ControlNetModel
- https://huggingface.co/docs/diffusers/v0.26.3/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline
- https://huggingface.co/docs/diffusers/v0.26.3/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline
- https://huggingface.co/docs/diffusers/v0.26.3/en/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline
- https://huggingface.co/docs/diffusers/using-diffusers/callback

pip install peft
peft-0.9.0
'''

###########################################################
# Libraries
###########################################################
import os
import gc
import torch
from PIL import Image

###########################################################
# Diffusers 0.26.3
###########################################################
from diffusers import(
# StableDiffusion 1.5 Models

ControlNetModel,
StableDiffusionControlNetPipeline,
StableDiffusionControlNetImg2ImgPipeline,
StableDiffusionImg2ImgPipeline,

# StableDiffusion XL Models

StableDiffusionXLPipeline,
StableDiffusionXLImg2ImgPipeline,
StableDiffusionXLControlNetPipeline,
StableDiffusionXLControlNetImg2ImgPipeline,

# Schedulers

AutoencoderKL,
EulerDiscreteScheduler,
EulerAncestralDiscreteScheduler,
HeunDiscreteScheduler,
UniPCMultistepScheduler,
DDIMScheduler,
DDPMScheduler,
DPMSolverMultistepScheduler,
)

from transformers import CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from diffusers.models.attention_processor import AttnProcessor2_0

# Image Encoder
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "models/ip_adapter/h94/IP_Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to("cuda")

# VaeImageProcessor
control_image_processor = VaeImageProcessor(
    do_resize=True,
    vae_scale_factor=8,
    resample="lanczos",
    do_normalize=True,
    do_binarize=False,
    do_convert_rgb=False,
    do_convert_grayscale=False,
)

###########################################################
# Models
###########################################################
# Variational autoencoder model
vae_model = "models/vae/sd-vae-ft-mse"
# ControlNet model
controlnet_model = "models/controlnet_models/control_v11f1e_sd15_tile"
# Stable Diffusion 1.5 model
sd15_model = "models/sd15/photoRealV15_photorealv21"
# sd15_model = "models/sd15/juggernaut_v17"

# Save Images
output_directory = "images/output_images"
###########################################################
# Input
###########################################################
input_image = load_image(image="images/input_image_set/test_03_512.png", convert_method=lambda img: img.convert("RGB")) # input image to be upscaled
ip_adapter_image = load_image(image="images/input_image_set/test_03_512.png", convert_method=lambda img: img.convert("RGB")) # image of transformation

# Checking Input Image
print(input_image)
original_width, original_height = input_image.size[0], input_image.size[1]
print(original_width, original_height)

# Checking IP Adapter Image
print(ip_adapter_image)
print(ip_adapter_image.size[0], ip_adapter_image.size[1])

# vae for SD1.5 models
vae = AutoencoderKL.from_pretrained(vae_model,
                                    torch_dtype=torch.float16).to("cuda")
gc.collect()
torch.cuda.empty_cache()

# ControlNetModel
controlnet = ControlNetModel.from_pretrained(controlnet_model,
                                             torch_dtype=torch.float16).to("cuda")
gc.collect()
torch.cuda.empty_cache()

###########################################################
# Stable Diffusion Pipelines
###########################################################

# StableDiffusionImg2ImgPipeline
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    sd15_model,
    vae=vae,
    safety_checker=None,
    use_safetensors=True,
    torch_dtype=torch.float16
).to("cuda")
gc.collect()
torch.cuda.empty_cache()

# Scheduler with StableDiffusionImg2ImgPipeline
pipe_img2img.scheduler = EulerDiscreteScheduler.from_config(pipe_img2img.scheduler.config)

# StableDiffusionControlNetPipeline
pipe_ctrl = StableDiffusionControlNetPipeline.from_pretrained(
    sd15_model,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    use_safetensors=True,
    torch_dtype=torch.float16
).to("cuda")
gc.collect()
torch.cuda.empty_cache()

# StableDiffusionControlNetImg2ImgPipeline
pipe_ctrl_img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    sd15_model,
    controlnet=controlnet,
    vae=vae,
    image_encoder=image_encoder,
    safety_checker=None,
    use_safetensors=True,
    torch_dtype=torch.float16
).to("cuda")
gc.collect()
torch.cuda.empty_cache()


###########################################################
# Reduce Memory (Experimental)
###########################################################

# pipe_img2img.enable_model_cpu_offload()
# pipe_img2img.unet.to(memory_format=torch.channels_last)
# pipe_img2img.unet.set_attn_processor(AttnProcessor2_0())
# pipe_img2img.enable_xformers_memory_efficient_attention()

# pipe_ctrl_img.enable_model_cpu_offload()
# pipe_ctrl_img.unet.to(memory_format=torch.channels_last)
# pipe_ctrl_img.unet.set_attn_processor(AttnProcessor2_0())
# pipe_ctrl_img.enable_vae_slicing()
# pipe_ctrl_img.disable_vae_tiling()
# pipe_ctrl_img.enable_xformers_memory_efficient_attention()


# IP Adapter - StableDiffusionControlNetPipeline
pipe_ctrl.load_ip_adapter("models/ip_adapter/h94/IP_Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_ctrl.set_ip_adapter_scale(0.80) # 0.80

# Lora - StableDiffusionControlNetPipeline
pipe_ctrl.load_lora_weights("models/lora/more_details.safetensors")
pipe_ctrl.fuse_lora(lora_scale=0.90) # 0.10 - 0.90 (variations can be introduced)

# FreeU - StableDiffusionControlNetPipeline
pipe_ctrl.enable_freeu(s1=0.90, s2=0.20, b1=1.50, b2=1.60) # s1=0.90, s2=0.20, b1=1.50, b2=1.60

# Scheduler with StableDiffusionControlNetPipeline
# pipe_ctrl.scheduler = DDIMScheduler.from_config(pipe_ctrl.scheduler.config)
pipe_ctrl.scheduler = DDIMScheduler.from_config(pipe_ctrl.scheduler.config)

# Reduce Memory
pipe_ctrl.enable_model_cpu_offload()
pipe_ctrl.unet.to(memory_format=torch.channels_last)
pipe_ctrl.enable_vae_slicing()
pipe_ctrl.enable_vae_tiling()
# Do not enable if using IP Adapter
# pipe_ctrl.unet.set_attn_processor(AttnProcessor2_0())
# pipe_ctrl.enable_xformers_memory_efficient_attention()

# IP Adapter - StableDiffusionControlNetPipeline
pipe_ctrl_img.load_ip_adapter("models/ip_adapter/h94/IP_Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_ctrl_img.set_ip_adapter_scale(0.80) # 0.80


###########################################################
# Step I
###########################################################
# StableDiffusionControlNetPipeline HyperParameters
prompt = "photorealistic masterpiece, best high quality, incredibly highly detailed, sharp focus, 8k 50mm resolution"
negative_prompt = "poorly drawn, ugly, tiling, out of frame, mutation, mutated, extra fingers, too many fingers, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad hands, bad anatomy, blurred, text, watermark, grainy,  writing, calligraphy, sign, cut off"
num_inference_steps = 32
guidance_scale = 7.5
controlnet_conditioning_scale = 1.0
controlnet_guidance_start = 0.4
controlnet_guidance_end = 1.0
generator_seed = 123456789
width = int(original_width * 1.25)
height = int(original_height * 1.25)
output_type = "latent"

# Prompt_embeds
max_length = pipe_ctrl.tokenizer.model_max_length
input_ids = pipe_ctrl.tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")
negative_ids = pipe_ctrl.tokenizer(negative_prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").input_ids
negative_ids = negative_ids.to("cuda")
concat_embeds = []
neg_embeds = []
for i in range(0, input_ids.shape[-1], max_length):
    concat_embeds.append(pipe_ctrl.text_encoder(input_ids[:, i: i + max_length])[0])
    neg_embeds.append(pipe_ctrl.text_encoder(negative_ids[:, i: i + max_length])[0])

prompt_embeds = torch.cat(concat_embeds, dim=1)
negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

# Generate image using StableDiffusionControlNetPipeline
gc.collect()
torch.cuda.empty_cache()
pipe_ctrl_image = pipe_ctrl(
    # prompt=prompt,
    # negative_prompt=negative_prompt,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    image=input_image,
    ip_adapter_image=ip_adapter_image,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    controlnet_guidance_start=controlnet_guidance_start,
    controlnet_guidance_end=controlnet_guidance_end,
    # cross_attention_kwargs={"scale": lora_scale},
    generator=torch.manual_seed(generator_seed),
    output_type=output_type
).images
gc.collect()
torch.cuda.empty_cache()
print(pipe_ctrl_image)

# Decoding the latents
with torch.no_grad():
    image = pipe_ctrl.decode_latents(pipe_ctrl_image)
image = pipe_ctrl.numpy_to_pil(image)[0]
image.save(os.path.join(output_directory, 'pipe_ctrl_image.jpg'), format="JPEG", quality=90)

###########################################################
# Step II
###########################################################

# Dynamic CFG
def callback_dynamic_cfg(pipe_img2img, step_index, timestep, callback_kwargs):
    max_length = pipe_img2img.tokenizer.model_max_length
    input_ids = pipe_img2img.tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length,
                                       return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")

    concat_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipe_img2img.text_encoder(input_ids[:, i: i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)

    # adjust the batch_size of prompt_embeds according to guidance_scale
    if step_index == int(pipe_img2img.num_timesteps * 0.4):
        print(pipe_img2img._guidance_scale)
        prompt_embeds = callback_kwargs["prompt_embeds"]

    # update guidance_scale and prompt_embeds
    pipe_img2img._guidance_scale = 0.0
    callback_kwargs["prompt_embeds"] = prompt_embeds
    return callback_kwargs

# Interrupt the diffusion process
# def interrupt_callback(pipe, i, t, callback_kwargs):
#     stop_idx = 10
#     if i == stop_idx:
#         pipe._interrupt = True
#
#     return callback_kwargs

# pipe(
#     "A photo of a cat",
#     num_inference_steps=num_inference_steps,
#     callback_on_step_end=interrupt_callback,
# )

prompt_1 = ""
prompt_2 = "incredibly highly detailed, best quality, 8k, high resolution"
prompt = prompt_1 + ", " + prompt_2

# Generate image using StableDiffusionImg2ImgPipeline
gc.collect()
torch.cuda.empty_cache()
pipe_img2img_image = pipe_img2img(
             prompt,
             negative_prompt_embeds=negative_prompt_embeds,
             image=pipe_ctrl_image,
             num_inference_steps=32,
             guidance_scale=7.5,
             strength=0.75,
             generator=torch.manual_seed(0),
             output_type="latent",
             callback_on_step_end=callback_dynamic_cfg,
             callback_on_step_end_tensor_inputs=['prompt_embeds']
             ).images
# print(pipe_img2img._guidance_scale)
# print(pipe_img2img._num_timesteps)

gc.collect()
torch.cuda.empty_cache()

# Decoding the latents
with torch.no_grad():
    image = pipe_img2img.decode_latents(pipe_img2img_image)
image = pipe_img2img.numpy_to_pil(image)[0]
image.save(os.path.join(output_directory, 'pipe_img2img_image.jpg'), format="JPEG", quality=90)

###########################################################
# Step III
###########################################################

# Generate image using StableDiffusionControlNetPipeline
gc.collect()
torch.cuda.empty_cache()
pipe_ctrl_img_image = pipe_ctrl_img(
    # prompt=prompt,
    # negative_prompt=negative_prompt,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    image=input_image,
    control_image=input_image,
    ip_adapter_image=input_image,
    width=width,
    height=height,
    strength=0.40,
    num_inference_steps=50,
    guidance_scale=4,
    controlnet_conditioning_scale=0.65,
    controlnet_guidance_start=controlnet_guidance_start,
    controlnet_guidance_end=controlnet_guidance_end,
    # cross_attention_kwargs={"scale": lora_scale},
    generator=torch.manual_seed(generator_seed),
    output_type=output_type
).images
gc.collect()
torch.cuda.empty_cache()
print(pipe_ctrl_img_image)

# Decoding the latents
with torch.no_grad():
    image = pipe_ctrl.decode_latents(pipe_ctrl_img_image)
image = pipe_ctrl.numpy_to_pil(image)[0]
image.save(os.path.join(output_directory, 'pipe_ctrl_img_image.jpg'), format="JPEG", quality=90)

###########################################################
# Step IV - Clean and Empty
###########################################################

# Clean
gc.collect()
torch.cuda.empty_cache()

# Unload
pipe_ctrl.unload_ip_adapter()
pipe_ctrl.unfuse_lora()
pipe_ctrl.disable_freeu()

# Delete
del vae
del controlnet
del pipe_ctrl
del pipe_ctrl_img
del pipe_img2img
gc.collect()
torch.cuda.empty_cache()