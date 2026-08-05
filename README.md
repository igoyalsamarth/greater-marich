# greater-marich

## Commands

```bash
uv run python main.py download video <url>
uv run python main.py download video <url> -o ./downloads
uv run python main.py download video <url> --format "best"
uv run python main.py convert audio <video>
uv run python main.py convert audio <video> -o audio/
uv run python main.py separate run <audio>
uv run python main.py separate run audio/video1.wav
uv run python main.py stt transcribe <audio>
uv run python main.py stt transcribe <audio> -o stt/
uv run python main.py translate diarized <stt-json>
uv run python main.py translate diarized <stt-json> -o translate/
uv run python main.py speech generate <translated-json>
uv run python main.py speech generate <translated-json> -o speech/video1/
uv run python main.py dub create speech/video1/video1.json
```
