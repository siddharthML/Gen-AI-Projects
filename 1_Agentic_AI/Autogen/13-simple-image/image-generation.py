"""
This script facilitates the generation of images based on textual prompts using
the HuggingFace API and Stable Diffusion model. An image is created in response
to the provided prompt and saved locally. The script includes necessary API 
setup, configuration, and agent workflow using the `autogen` library.
"""

import requests
import random
import io
import autogen
from PIL import Image
from typing import Annotated


# API URL and headers for authorization
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer your_token"}


def create_image(message: Annotated[str, "The response from the LLM"]) -> str:
    """
    Create an image from a given text prompt message and save to a file.

    Args:
        message (str): The text prompt for image generation.

    Returns:
        str: The original text prompt message.
    """
    # Send request to the API for image generation
    response = requests.post(API_URL, headers=headers, json=message)
    image_bytes = response.content

    # Generate a random file name for the image
    random_number = random.randint(1, 1000000)
    file_name = f"filename_{random_number}.png"

    # Save the generated image to a file
    Image.open(io.BytesIO(image_bytes)).save(file_name)
    return message


# Configuration for the language model
llm_config = {
    "config_list": autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json"),
    "temperature": 0.5,
    "seed": 41
}

# Initialize the assistant agent with system message and configuration
assistant = autogen.AssistantAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)

# Initialize the user proxy agent with termination condition and code execution config
user_proxy = autogen.UserProxyAgent(
    name="User",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

# Register the create_image function for LLM and execution
assistant.register_for_llm(name="create_image", description="Create an image from a text description")(create_image)
user_proxy.register_for_execution(name="create_image")(create_image)

# Initiate chat with the assistant agent to generate an image based on the prompt
user_proxy.initiate_chat(assistant, message="Create an image of a professional futbol player.")