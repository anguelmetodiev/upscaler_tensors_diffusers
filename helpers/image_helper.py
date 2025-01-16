# Image Helper

from PIL import Image

# Resize Images
def resize_images(image, scale_factor):
    image = image.convert("RGB")
    original_width, original_height = image.size
    base_dimensions = (original_width, original_height)
    new_dimensions = tuple(int(dim * scale_factor) for dim in base_dimensions)
    resized_image = image.resize(new_dimensions, Image.LANCZOS)
    return resized_image