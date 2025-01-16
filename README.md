# Upscaler Diffusers

```
pip3 -U xformers install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```
pip install diffusers transformers accelerate invisible-watermark opencv-python pytorch-lightning einops 
```

Version references:

```
accelerate               0.27.2
diffusers                0.26.3
einops                   0.7.0
invisible-watermark      0.2.0
opencv-python            4.9.0.80
pytorch-lightning        2.2.0.post0
safetensors              0.4.2
torch                    2.2.0+cu121
torchaudio               2.2.0+cu121
torchvision              0.17.0+cu121
transformers             4.38.1
xformers                 0.0.24
```

**Create the following folders**
```
mkdir -p models/{controlnet_models,ip_adapter,lora,open_models,upscaling_models,vae,sd15,sdxl,sdxl_turbo}
```

```
mkdir -p images/{input_images,output_images}
```

**Upscaler versions:**
- Closer to the original input image

- https://imgsli.com/MjQyODgx

