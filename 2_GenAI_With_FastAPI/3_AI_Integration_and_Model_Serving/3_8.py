from typing import Literal
from PIL import Image
from io import BytesIO

def img_to_bytes(
        image: Image.Image, img_format: Literal["PNG", "JPEG"] = "PNG"
) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=img_format)
    return buffer.getvalue()


from fastapi import FastAPI, Response, status
from model_vision import load_image_model, generate_image

app = FastAPI()

# For whatever reason this decorator doesnt work
# @app.get("/generate/image",
#          response={status.HTTP_200_OK: {"content": {"image/png": {}}}},
#          response_class=Response)

@app.get("/generate/image", response_class=Response)

def serve_text_to_image_model_controller(prompt: str):
    pipe = load_image_model()
    output = generate_image(pipe, prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")

