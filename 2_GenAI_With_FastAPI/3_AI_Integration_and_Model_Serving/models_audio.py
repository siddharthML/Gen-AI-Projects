# This module provides functionality to load a Bark audio model and generate audio
# from text prompts using specified voice presets.


from typing import Literal
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel, BarkProcessor, BarkModel

# Specify supported voice options using Literal
VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"]

# Determine the device to run the model on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    """
    Load the Bark model and its processor from pretrained resources.

    Returns:
        tuple[BarkProcessor, BarkModel]: The loaded processor and model.
    """
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")
    model.to(device)  # Ensure the model runs on the correct device
    return processor, model

def generate_audio(
    processor: BarkProcessor,
    model: BarkModel,
    prompt: str,
    preset: VoicePresets,
) -> tuple[np.ndarray, int]:
    """
    Generate audio from a given text prompt using the specified voice preset.

    Args:
        processor (BarkProcessor): The processor for handling input text.
        model (BarkModel): The model for generating audio.
        prompt (str): The text prompt to convert to audio.
        preset (VoicePresets): The voice preset to use for generation.

    Returns:
        tuple[np.ndarray, int]: The generated audio waveform and its sample rate.
    """
    # Process the text prompt into model input format with the specified voice preset
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        voice_preset=preset
    )

    # Explicitly set the attention mask to all ones
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to(device)
    
    # Move all input tensors to the correct device (GPU if available)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the audio waveform using the model and move the result to CPU
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
    
    # Retrieve the sample rate from the model's generation configuration
    sample_rate = model.generation_config.sample_rate
    
    return output, sample_rate