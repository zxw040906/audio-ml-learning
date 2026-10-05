"""Shared paths for repository data and generated outputs."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AUDIO_LEARNING_DATA", REPO_ROOT / ".local-data")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("AUDIO_LEARNING_OUTPUT", REPO_ROOT / ".local-output")).expanduser()
URBANSOUND8K_ROOT = Path(
    os.environ.get("URBANSOUND8K_ROOT", DATA_ROOT / "urbansound8k" / "UrbanSound8K")
).expanduser()
YESNO_AUDIO_DIR = REPO_ROOT / "learning" / "audio-deep-learning" / "data" / "yesno" / "waves_yesno"
SPEECH_AUDIO_DIR = REPO_ROOT / "learning" / "speech-signal-processing" / "assets" / "audio"


def output_dir(project, *parts):
    """Return and create a local output directory (ignored by Git)."""
    path = OUTPUT_ROOT / project
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path
