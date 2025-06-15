"""
This module serves as a BentoML service for image generation using a pre-trained model.
"""

import bentoml
import sys
import os

# Append the module's directory to sys.path
sys.path.append(os.path.dirname(__file__))

# Import the function to load the image model
from model_vision import load_image_model

# Define the BentoML service with specified resources and configuration
@bentoml.service(
    resources={"cpu": "4"}, traffic={"timeout": 120}, http={"port": 5000}
)
class Generate:
    def __init__(self) -> None:
        """
        Initialize the Generate service with the pre-trained image generation model.
        """
        # Load image generation pipeline from model_vision
        self.pipe = load_image_model()

    @bentoml.api(route="/generate/image")
    def generate(self, prompt: str) -> str:
        """
        Generate an image based on the provided prompt.

        Args:
            prompt (str): The text prompt for image generation.

        Returns:
            str: The generated image.
        """
        # Run the image generation pipeline with the provided prompt
        output = self.pipe(prompt, num_inference_steps=10).images[0]
        return output