from fastrtc import ReplyOnPause, Stream, AdditionalOutputs
import numpy as np
from numpy.typing import NDArray

from dotenv import load_dotenv
from groq import Groq
from elevenlabs import ElevenLabs

import os
import time
import tempfile
import wave
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def audio_to_wav_file(audio_data, sample_rate):
    """Convert audio data to a temporary WAV file for Groq processing"""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return temp_file.name


def transcribe_with_groq(audio):
    """Transcribe audio using Groq's Whisper API"""
    sample_rate, audio_data = audio

    # Convert audio data to a temporary WAV file
    wav_file = audio_to_wav_file(audio_data, sample_rate)

    try:
        with open(wav_file, "rb") as file:
            translation = groq_client.audio.translations.create(
                file=(wav_file, file.read()),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0
            )
            return translation.text
    finally:
        # Clean up the temporary file
        if os.path.exists(wav_file):
            os.remove(wav_file)


def generate_response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    chatbot = chatbot or []
    messages = [{"role": msg["role"], "content": msg["content"]}
                for msg in chatbot]
    start = time.time()
    text = transcribe_with_groq(audio)
    print("transcription", time.time() - start)
    print("prompt", text)

    chatbot.append({"role": "user", "content": text})

    yield AdditionalOutputs(chatbot)

    messages.append({"role": "user", "content": text})
    response_text = (
        groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=512,
            messages=messages,
        )
        .choices[0]
        .message.content
    )

    chatbot.append({"role": "assistant", "content": response_text})

    yield response_text


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    # Transcription and response generation
    gen = generate_response(audio, chatbot)

    # First yield is AdditionalOutputs with updated chatbot
    chatbot = next(gen)

    # Second yield is the response text
    response_text = next(gen)

    print(response_text)

    # Fall back to ElevenLabs for reliable TTS
    print("Using ElevenLabs for text-to-speech")
    for chunk in tts_client.text_to_speech.convert_as_stream(
        text=response_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
    ):
        audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        yield (24000, audio_array)


stream = Stream(
    handler=ReplyOnPause(response, input_sample_rate=16000),
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "LLM Voice Chat (Powered By Groq, ElevenLabs, and WebRTC ⚡️)"},
)

stream.ui.launch()
