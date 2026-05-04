"""
Real-Time Speech Assistant
Pipeline: Microphone → Whisper (STT) → GPT-4o-mini (LLM) → ElevenLabs (TTS) → Speaker

Requirements:
    pip install openai elevenlabs sounddevice soundfile numpy python-dotenv pyaudio

Environment variables (.env):
    OPENAI_API_KEY=sk-...
    ELEVENLABS_API_KEY=...
    ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM   # default: Rachel voice
"""

import os
import sys
import time
import wave
import tempfile
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

# ─── Clients ────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

# ─── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000       # Whisper works best at 16kHz
CHANNELS = 1
DTYPE = np.int16
SILENCE_THRESHOLD = 500   # RMS threshold to detect silence
SILENCE_DURATION = 1.5    # seconds of silence before stopping recording
MIN_RECORDING_DURATION = 0.5  # minimum seconds before checking for silence

SYSTEM_PROMPT = """You are a helpful, concise voice assistant. 
Keep responses conversational and brief (2-4 sentences max) since they'll be spoken aloud.
Be warm, natural, and direct. Avoid markdown, lists, or special formatting."""

# ─── Conversation history ────────────────────────────────────────────────────
conversation_history = []


def rms(audio_chunk: np.ndarray) -> float:
    """Root mean square — measures audio loudness."""
    return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))


def record_until_silence() -> np.ndarray | None:
    """
    Records audio from the microphone.
    Stops automatically after SILENCE_DURATION seconds of quiet.
    Returns numpy array of audio samples, or None if nothing was recorded.
    """
    print("\n🎤 Listening... (speak now, pause to stop)")

    audio_chunks = []
    silent_chunks = 0
    recording_started = False
    start_time = time.time()

    chunk_duration = 0.1  # seconds per chunk
    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    silence_chunks_needed = int(SILENCE_DURATION / chunk_duration)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=chunk_samples,
        ) as stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                chunk_flat = chunk.flatten()
                chunk_rms = rms(chunk_flat)

                audio_chunks.append(chunk_flat)

                elapsed = time.time() - start_time

                # Start tracking silence only after minimum recording time
                if elapsed >= MIN_RECORDING_DURATION:
                    if chunk_rms < SILENCE_THRESHOLD:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                        recording_started = True

                    if recording_started and silent_chunks >= silence_chunks_needed:
                        break

                # Safety max duration (30s)
                if elapsed > 30:
                    print("⚠️  Max recording time reached.")
                    break

    except KeyboardInterrupt:
        return None

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks)

    # Check if anything meaningful was recorded
    if rms(audio) < SILENCE_THRESHOLD * 0.5:
        print("🔇 No speech detected.")
        return None

    print(f"✅ Recorded {len(audio) / SAMPLE_RATE:.1f}s of audio")
    return audio


def transcribe(audio: np.ndarray) -> str | None:
    """
    Converts audio to text using OpenAI Whisper.
    Saves audio to a temp WAV file, sends to Whisper API.
    """
    print("🔄 Transcribing...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
            )

        text = transcript.text.strip()
        if text:
            print(f"📝 You said: \"{text}\"")
        return text if text else None

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def get_llm_response(user_text: str) -> str | None:
    """
    Sends the transcribed text to GPT-4o-mini and returns the response.
    Maintains conversation history for multi-turn dialogue.
    """
    print("🧠 Thinking...")

    conversation_history.append({"role": "user", "content": user_text})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation_history,
            ],
            temperature=0.7,
            max_tokens=200,
        )

        reply = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": reply})

        # Keep history manageable (last 10 turns = 20 messages)
        if len(conversation_history) > 20:
            conversation_history.pop(0)
            conversation_history.pop(0)

        print(f"💬 Assistant: \"{reply}\"")
        return reply

    except Exception as e:
        print(f"❌ LLM error: {e}")
        return None


def speak(text: str) -> None:
    """
    Converts text to speech using ElevenLabs and plays it.
    Uses streaming for lower latency on longer responses.
    """
    print("🔊 Speaking...")

    try:
        audio = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2",  # lowest latency model
            output_format="mp3_44100_128",
        )
        play(audio)

    except Exception as e:
        print(f"❌ TTS error: {e}")
        # Fallback: print the response so the user still gets it
        print(f"   (TTS failed — response was: {text})")


def check_exit_command(text: str) -> bool:
    """Returns True if the user said something like 'goodbye' or 'exit'."""
    exit_phrases = ["goodbye", "bye", "exit", "quit", "stop", "that's all", "see you"]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in exit_phrases)


def run():
    """Main loop — listens, transcribes, responds, speaks. Repeat."""
    print("=" * 55)
    print("  Real-Time Speech Assistant")
    print("  Powered by Whisper + GPT-4o-mini + ElevenLabs")
    print("=" * 55)
    print("  Say 'goodbye' or press Ctrl+C to exit.\n")

    # Greeting
    greeting = "Hello! I'm your voice assistant. How can I help you today?"
    print(f"💬 Assistant: \"{greeting}\"")
    speak(greeting)

    try:
        while True:
            # Step 1: Record
            audio = record_until_silence()
            if audio is None:
                continue

            # Step 2: Transcribe
            user_text = transcribe(audio)
            if not user_text:
                continue

            # Step 3: Check for exit
            if check_exit_command(user_text):
                farewell = "Goodbye! Have a great day!"
                print(f"💬 Assistant: \"{farewell}\"")
                speak(farewell)
                break

            # Step 4: LLM response
            reply = get_llm_response(user_text)
            if not reply:
                continue

            # Step 5: Speak
            speak(reply)

    except KeyboardInterrupt:
        print("\n\n👋 Exiting. Goodbye!")

if __name__ == "__main__":
    run()
