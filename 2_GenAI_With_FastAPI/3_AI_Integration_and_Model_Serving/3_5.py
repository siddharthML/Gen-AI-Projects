# FastAPI endpoint for returning generated audio
# Run using -  uvicorn 3_5:app --reload

from io import BytesIO
import soundfile
import numpy as np
import functools
from fastapi import FastAPI, status, Response
from fastapi.responses import StreamingResponse
from models_audio import load_audio_model, generate_audio, VoicePresets

def audio_array_to_buffer(audio_array: np.ndarray, sample_rate: int) -> BytesIO:
    """
    Converts the generated audio array into a buffer compatible with WAV format.

    Parameters:
    audio_array (np.ndarray): The generated audio as a NumPy array.
    sample_rate (int): The sample rate of the audio.

    Returns:
    BytesIO: A buffer containing the audio in WAV format.
    """
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, samplerate=sample_rate, format="wav")
    buffer.seek(0)  # Ensure the buffer pointer is reset to the beginning
    return buffer

# Initialize the FastAPI app
app = FastAPI()

# Cache the loaded model to avoid reloading on every request
@functools.lru_cache()
def get_cached_audio_model():
    """
    Loads and caches the audio generation model.

    Returns:
    Tuple: A tuple containing the processor and the model.
    """
    return load_audio_model()

@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1",
):
    """
    Endpoint to generate and return audio based on text prompt.

    Parameters:
    prompt (str): The text prompt for audio generation.
    preset (VoicePresets, optional): Voice preset for generation, defaults to "v2/en_speaker_1".

    Returns:
    StreamingResponse: A streamed response containing the generated audio in WAV format.
    """
    processor, model = get_cached_audio_model()
    
    try:
        output, sample_rate = generate_audio(processor, model, prompt, preset)
        return StreamingResponse(audio_array_to_buffer(output, sample_rate), media_type="audio/wav")
    except Exception as e:
        print(f"Error in streaming response: {e}")
        return Response(status_code=500, content=f"Error generating audio: {str(e)}")