# This module provides functionality to load a Stable Diffusion image generation model
# and generate images from text prompts using that model.

import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipelineLegacy
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_model() -> StableDiffusionInpaintPipelineLegacy:
    """
    Load the image generation model.

    This function loads a pre-trained diffusion pipeline model for image generation.
    The model is loaded onto the available device (GPU if available, otherwise CPU).

    Returns:
        StableDiffusionInpaintPipelineLegacy: The loaded diffusion pipeline model.
    """
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", torch_dtype=torch.float32, device=device
    )
    return pipe

def generate_image(
    pipe: StableDiffusionInpaintPipelineLegacy, prompt: str
) -> Image.Image:
    """
    Generate an image based on a text prompt.

    This function uses the provided diffusion pipeline model to generate an image
    based on the given text prompt. The number of inference steps is set to 10.

    Args:
        pipe (StableDiffusionInpaintPipelineLegacy): The diffusion pipeline model.
        prompt (str): The text prompt to generate the image.

    Returns:
        Image.Image: The generated image.
    """
    output = pipe(prompt, num_inference_steps=10).images[0]
    return output