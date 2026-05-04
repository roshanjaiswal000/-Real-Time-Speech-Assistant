# Real-Time Speech Assistant

An end-to-end voice AI pipeline built with Python.

**Pipeline:** Microphone → Whisper (speech-to-text) → GPT-4o-mini (response) → ElevenLabs or OpenAI TTS (text-to-speech) → Speaker

---

## Features

- **Auto silence detection** — starts and stops recording automatically, no button press needed
- **Multi-turn memory** — remembers the last 10 exchanges in the conversation
- **Two versions** — ElevenLabs (higher quality voice) or OpenAI TTS only (simpler setup)
- **Graceful exit** — just say "goodbye" to end the session
- **Error resilience** — TTS failures fall back to printed text so you never lose a response

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd speech_assistant
pip install -r requirements.txt
```

> **macOS users:** You may need PortAudio first:
> ```bash
> brew install portaudio
> ```

> **Ubuntu/Debian users:**
> ```bash
> sudo apt-get install portaudio19-dev python3-dev
> ```

### 2. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:
- `OPENAI_API_KEY` — required for both Whisper and GPT-4o-mini
- `ELEVENLABS_API_KEY` — optional, for higher-quality TTS

### 3. Run

**With ElevenLabs TTS (recommended):**
```bash
python assistant.py
```

**OpenAI TTS only (no ElevenLabs needed):**
```bash
python assistant_openai_only.py
```

---

## How it works

### 1. Recording (`record_until_silence`)
Uses `sounddevice` to stream audio from the microphone in 100ms chunks. Calculates RMS (root mean square) loudness per chunk. After the user finishes speaking, silence is detected for 1.5 seconds and recording stops automatically.

### 2. Transcription (`transcribe`)
Saves the raw audio to a temporary WAV file, sends it to OpenAI Whisper via the API, and returns the transcript text.

### 3. LLM response (`get_llm_response`)
Appends the user's message to the conversation history and sends the full history (plus a system prompt) to GPT-4o-mini. The system prompt instructs the model to keep responses short and conversational for audio output.

### 4. Text-to-speech (`speak`)
- **ElevenLabs version:** Uses `eleven_turbo_v2` model for low latency, streams audio bytes and plays with `elevenlabs.play()`
- **OpenAI version:** Uses `tts-1` with `pcm` output format (avoids MP3 decode step), plays raw PCM directly via `sounddevice`

---

## Customization

### Change the voice (ElevenLabs)
Browse voices at https://elevenlabs.io/voice-library, copy the voice ID, and set `ELEVENLABS_VOICE_ID` in your `.env`.

### Change the voice (OpenAI)
In `assistant_openai_only.py`, change `TTS_VOICE` to one of: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`.

### Change the assistant's personality
Edit the `SYSTEM_PROMPT` constant in either file. Keep it short — this gets sent with every request.

### Adjust silence detection sensitivity
- `SILENCE_THRESHOLD` — raise if it cuts off mid-sentence in noisy environments, lower if it waits too long after you stop speaking
- `SILENCE_DURATION` — seconds of quiet before stopping (default 1.5s)

---

## Resume bullet point

> "Built a real-time voice AI pipeline using OpenAI Whisper for speech recognition, GPT-4o-mini for conversational responses, and ElevenLabs for text-to-speech. Implemented RMS-based silence detection for hands-free operation and maintained multi-turn conversation history. Full Python implementation deployed locally."

---

## Estimated API costs

| Component | Model | Cost |
|---|---|---|
| Whisper STT | whisper-1 | $0.006 / minute |
| LLM | gpt-4o-mini | ~$0.0003 / response |
| TTS | OpenAI tts-1 | $0.015 / 1K characters |
| TTS | ElevenLabs turbo | ~$0.18 / 1K characters |

A typical 10-minute conversation costs roughly $0.10–$0.30 total.
