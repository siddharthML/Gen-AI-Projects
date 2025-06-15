"""
This module provides functionality for loading a text generation model and generating text using the FastAPI framework.
It is specifically designed to simulate a chatbot that teaches users how to set up and use FastAPI.
"""

import torch
from transformers import Pipeline, pipeline

# Define prompt for user query
prompt = "How to set up a FastAPI project?"

# Define system prompt for chatbot behavior
system_prompt = """
Your name is FastAPI bot and you are a helpful
chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
"""

# Determine the device (GPU if available, else CPU) for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_model() -> Pipeline:
    """
    Loads the text generation model using transformers pipeline.

    Returns:
        Pipeline: A text generation pipeline object.
    """
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    return pipe


def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    """
    Generates text based on the provided prompt using the specified pipeline.

    Args:
        pipe (Pipeline): The text generation pipeline object.
        prompt (str): The user query prompt.
        temperature (float, optional): Sampling temperature for text generation. Default is 0.7.

    Returns:
        str: Generated text response from the model.
    """
    # Prepare messages for the model, including system and user prompts
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Tokenize and format the prompt message for the model
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate text using the model
    predictions = pipe(
        prompt,
        temperature=temperature,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    # Extract and return the generated text from the model's output
    output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
    return output