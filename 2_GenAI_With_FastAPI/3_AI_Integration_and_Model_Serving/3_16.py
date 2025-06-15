"""
This module demonstrates model preloading with a FastAPI application lifespan.
Model preloading trades off RAM for service efficiency by loading the model once
instead of for every request.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Response, status
from model_vision import load_image_model, generate_image

# from utils import img_to_bytes # this is in 3_8.py

# Initialize a global dictionary to store models
models = {}

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager to handle the start-up and shutdown lifecycle of the FastAPI app.
    Loads the 'text2image' model at startup and releases resources at shutdown.
    """
    models["text2image"] = load_image_model()
    yield
    # Run cleanup code here
    models.clear()

# Create the FastAPI application and assign the lifespan context manager
app = FastAPI(lifespan=lifespan)

@app.get("/generate/image", response_class=Response)
def serve_text_to_image_model_controller(prompt: str):
    """
    Endpoint to generate an image from the prompt using the preloaded 'text2image' model.
    
    Args:
        prompt (str): The text prompt to generate an image.

    Returns:
        Response: An image in PNG format.
    """
    output = generate_image(models["text2image"], prompt=prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")


