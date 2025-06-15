# Serving a language model via a FastAPI endpoint
# main.py
# Run using uvicorn 3_2:app --reload

from fastapi import FastAPI
from models_text import load_text_model, generate_text

app = FastAPI()

# Endpoint to generate text based on a given prompt
@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    """
    Generate text based on the provided prompt using a pre-trained language model.

    Args:
        prompt (str): The input prompt for the language model.

    Returns:
        str: The generated text from the language model.
    """
    # Load the pre-trained text generation model
    pipe = load_text_model()
    
    # Generate text using the loaded model and the provided prompt
    output = generate_text(pipe, prompt)
    
    # Return the generated text
    return output