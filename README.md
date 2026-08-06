# greater-marich

## Setup

Add these to your `.env`:

```bash
SARVAM_API_KEY=...
OPENAI_API_KEY=...  # for translation
HF_TOKEN=...  # Hugging Face token for pyannote embeddings and emotion2vec
```

Accept the user conditions for:
- [pyannote/embedding](https://huggingface.co/pyannote/embedding)

STT uses **Sarvam** for diarization and transcription (2 speakers), then enriches each turn with embeddings and emotion.

Separation uses a hybrid of two BS-RoFormer models: **jarredou SW** for vocals and **unwa vocals resurrection** for instrumental.

Translation uses a **two-step Pydantic AI** flow with **OpenAI** (`gpt-5.4-mini` by default):
1. Summarize the full scene (plot, characters, tone, likely STT fixes)
2. Translate each line with that summary plus surrounding dialogue context

Translation writes:
- `outputs/translate/<name>/dialogues.json` (same shape as STT dialogues, with translated transcripts)

Speech uses **Sarvam Bulbul** (`bulbul:v3` by default) on translated dialogues, with vocals volume matching per segment.

Speech writes:
- `outputs/speech/<name>/*.wav`
- `outputs/speech/<name>/<name>.json`

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
uv run python main.py dub create outputs/speech/video1/video1.json
```

STT writes:
- `outputs/stt/<name>/characters.json`
- `outputs/stt/<name>/dialogues.json`

### Emotion model download issues

Emotion2Vec+ large is ~2 GB. If a download is interrupted, Hugging Face can leave a corrupt cache and FunASR will fail with `model 'iic/emotion2vec_plus_large' is not registered`.

```bash
rm -rf ~/.cache/huggingface/hub/models--emotion2vec--emotion2vec_plus_large
uv run python main.py stt transcribe outputs/audio/video1.wav
```
