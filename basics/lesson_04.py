from fastrtc import ReplyOnPause, Stream, AdditionalOutputs, get_tts_model, KokoroTTSOptions
import numpy as np
from numpy.typing import NDArray

from dotenv import load_dotenv
from groq import Groq

import os
import time
import tempfile
import wave
import gradio as gr
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Use Kokoro for Text To Speech (TTS)
tts_client = get_tts_model(model="kokoro")


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
    voice: str = "af_heart",
    speed: float = 1.0,
    lang: str = "en-us",
    chatbot: list[dict] | None = None,
):
    # Transcription and response generation
    gen = generate_response(audio, chatbot)

    # First yield is AdditionalOutputs with updated chatbot
    chatbot = next(gen)

    # Second yield is the response text
    response_text = next(gen)

    print(response_text)

    # Use Kokoro for TTS
    tts_options = KokoroTTSOptions(
        voice=voice,
        speed=speed,
        lang=lang
    )
    for chunk in tts_client.stream_tts_sync(
        response_text, options=tts_options
    ):
        yield chunk

# Docs on available languages and voicess: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
stream = Stream(
    handler=ReplyOnPause(response, input_sample_rate=16000),
    modality="audio",
    mode="send-receive",
    additional_inputs=[
        gr.Dropdown(
            label="Voice",
            choices=[
                # 🇺🇸 American English
                "af_heart", "af_bella", "af_nicole", "am_fenrir", "am_puck",
                # 🇬🇧 British English
                "bf_emma", "bf_isabella", "bm_fable", "bm_george",
                # 🇯🇵 Japanese
                "jf_alpha", "jf_gongitsune", "jf_tezukuro", "jm_kumo",
                # 🇨🇳 Mandarin Chinese
                "zf_xiaobei", "zm_yunjian",
                # 🇪🇸 Spanish
                "ef_dora", "em_alex",
                # 🇫🇷 French
                "ff_siwis",
                # 🇮🇳 Hindi
                "hf_alpha", "hm_omega",
                # 🇮🇹 Italian
                "if_sara", "im_nicola",
                # 🇧🇷 Brazilian Portuguese
                "pf_dora", "pm_alex"
            ],
            value="af_heart",
        ),
        gr.Slider(
            label="Speed",
            minimum=0.5,
            maximum=2.0,
            step=0.1,
            value=1.0,
        ),
        gr.Dropdown(
            label="Language",
            choices=[
                "en-us", "en-uk", "ja", "zh", "es", "fr", "hi", "it", "pt-br"
            ],
            value="en-us",
        ),
    ],
    ui_args={
        "title": "LLM Voice Chat (Powered By Groq, Kokoro, and WebRTC ⚡️)"
    },
)

stream.ui.launch()
