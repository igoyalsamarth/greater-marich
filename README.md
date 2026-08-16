# greater-marich

## Setup

Add these to your `.env`:

```bash
SARVAM_API_KEY=...
OPENAI_API_KEY=...  # for translation
HF_TOKEN=...  # optional; used if Hugging Face gated models are needed
```

STT uses **Sarvam** for diarization and transcription (2 speakers), then enriches each turn with **WavLM-large** character analysis (gender, emotion, voice embedding) plus signal-based pitch, energy, and speech rate.

On first STT run, the Vox-Profile model wrappers are cloned to `~/.cache/greater-marich/vox-profile-release`. The fine-tuned heads (`tiantiaf/wavlm-large-age-sex`, `tiantiaf/wavlm-large-categorical-emotion`) load from Hugging Face; the shared `microsoft/wavlm-large` processor is loaded once, and each head is kept in memory only while it runs.

Separation uses a hybrid of two BS-RoFormer models: **jarredou SW** for vocals and **unwa vocals resurrection** for instrumental.

Translation uses a **two-step Pydantic AI** flow with **OpenAI** (`gpt-5.4-mini` by default):
1. Summarize the full scene (plot, characters, emotional arc from per-line `emotion_profile`, tone, likely STT fixes)
2. Translate each line with that summary plus character `attributes`, line-level `emotion_profile`, and surrounding dialogue context

Translation writes:
- `outputs/translate/<name>/dialogues.json` (same shape as STT dialogues, with translated transcripts and preserved `emotion_profile` per line)

Speech uses **Sarvam Bulbul** (`bulbul:v3` by default) on translated dialogues, with vocals volume matching per segment.

Chatterbox speech (`speech-chatterbox`) uses **[ResembleAI/Chatterbox-Multilingual-hi](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-hi)** zero-shot cloning. For each speaker it concatenates all of their source vocal segments from the separated vocals track into one reference clip, then synthesizes each translated Hindi line with per-line `emotion_profile` mapped to Chatterbox exaggeration.

Install Chatterbox extras first:

```bash
uv sync --extra chatterbox
uv run python main.py speech-chatterbox download
```

On first run, model weights are downloaded from Hugging Face to `~/.cache/greater-marich/chatterbox-hi`.

Speech writes:
- `outputs/speech/<name>/*.wav` (Sarvam)
- `outputs/speech/<name>/<name>.json`
- `outputs/speech-chatterbox/<name>/*.wav` (Chatterbox Hindi)
- `outputs/speech-chatterbox/<name>/<name>.json`

## Commands

```bash
uv run python main.py download video <url>
uv run python main.py download video <url> -o outputs/videos
uv run python main.py convert audio <video>
uv run python main.py convert audio <video> -o outputs/audio/
uv run python main.py separate run outputs/audio/video1.wav
uv run python main.py stt transcribe outputs/audio/video1.wav
uv run python main.py translate diarized outputs/stt/video1/dialogues.json
uv run python main.py speech generate outputs/translate/video1/dialogues.json
uv run python main.py speech-chatterbox generate outputs/translate/video1/dialogues.json
uv run python main.py dub create outputs/speech-chatterbox/video1/video1.json
```

STT writes:
- `outputs/stt/<name>/characters.json` (`attributes` + `emotion_profile` per speaker)
- `outputs/stt/<name>/dialogues.json` (per-line `emotion_profile` on each diarized entry)
