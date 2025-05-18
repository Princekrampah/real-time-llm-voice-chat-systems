from fastrtc import (
    ReplyOnPause,
    Stream,
    AdditionalOutputs,
    get_stt_model, get_tts_model,
    KokoroTTSOptions
)
import numpy as np
from numpy.typing import NDArray

from dotenv import load_dotenv
from groq import Groq

import os
import time
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Text To Speech (TTS)
tts_client = get_tts_model(model="kokoro")
# Speech To Text (STT)
stt_model = get_stt_model(model="moonshine/base")

options = KokoroTTSOptions(
    voice="af_heart",
    speed=1.0,
    lang="en-us"
)


def generate_response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    chatbot = chatbot or []
    messages = [{"role": msg["role"], "content": msg["content"]}
                for msg in chatbot]
    start = time.time()

    # Use local STT model instead of Groq's Whisper
    text = stt_model.stt(audio)

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
    chatbot: list[dict] | None = None,
    tts_options: KokoroTTSOptions | None = None,
):
    # Transcription and response generation
    gen = generate_response(audio, chatbot)

    # First yield is AdditionalOutputs with updated chatbot
    chatbot = next(gen)

    # Second yield is the response text
    response_text = next(gen)

    print(response_text)

    # Use tts_client.stream_tts_sync for TTS (Local TTS)
    # Pass tts_options if provided, else use default options
    tts_options = KokoroTTSOptions(
        voice="bf_alice",
        speed=1.0,
        lang="en-us"
    )
    for chunk in tts_client.stream_tts_sync(
        response_text, options=tts_options or options
    ):
        yield chunk


stream = Stream(
    handler=ReplyOnPause(response, input_sample_rate=16000),
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "LLM Voice Chat (Powered by Groq, Kokoro, and Moonshine ⚡️)",
        "inputs": [
            {"name": "voice", "type": "dropdown", "choices": [
                "af_heart", "af_sun", "af_moon"], "label": "Voice", "value": "af_heart"},
            {"name": "speed", "type": "slider", "min": 0.5, "max": 2.0,
                "step": 0.1, "label": "Speed", "value": 1.0},
            {"name": "lang", "type": "dropdown", "choices": [
                "en-us", "en-uk"], "label": "Language", "value": "en-us"},
        ]
    },
)

stream.ui.launch()
