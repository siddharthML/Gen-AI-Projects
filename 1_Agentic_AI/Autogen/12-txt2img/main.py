"""
This script sends a payload to the Hugging Face Inference API to generate an image 
based on a given text prompt. The generated image is then saved locally using the PIL library.
"""

import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": "Bearer your_token"}

def query(payload):
    """
    Sends a POST request to Hugging Face Inference API with the given payload.

    Args:
        payload (dict): The JSON payload containing the text prompt.

    Returns:
        bytes: The content of the response, which is the generated image in bytes.
    """
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.content

def main():
    """
    Main function to execute the image generation and saving procedure.
    """
    # Generate image based on the text prompt
    image_bytes = query({"inputs": "Astronaut riding a horse"})

    # Open image from bytes and save it locally
    image = Image.open(io.BytesIO(image_bytes))
    image.save("test.png")

if __name__ == "__main__":
    main()