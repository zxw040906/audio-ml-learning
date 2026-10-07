"""Shared paths for repository data and generated outputs."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# 允许从任意工作目录直接执行 learning/ 下的脚本。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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
