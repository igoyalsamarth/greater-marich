import typer

from cli.convert import app as convert_app
from cli.download import app as download_app
from cli.dub import app as dub_app
from cli.separate import app as separate_app
from cli.speech import app as speech_app
from cli.stt import app as stt_app
from cli.translate import app as translate_app

app = typer.Typer(help="greater-marich CLI.")
app.add_typer(download_app, name="download")
app.add_typer(separate_app, name="separate")
app.add_typer(convert_app, name="convert")
app.add_typer(stt_app, name="stt")
app.add_typer(translate_app, name="translate")
app.add_typer(speech_app, name="speech")
app.add_typer(dub_app, name="dub")
