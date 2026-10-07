from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _import_path in (str(_REPO_ROOT), str(_Path(__file__).resolve().parent)):
    if _import_path not in _sys.path:
        _sys.path.insert(0, _import_path)

from learning.common.paths import (
    SPEECH_AUDIO_DIR,
)

import librosa
from matplotlib import pyplot as plt
import numpy as np

import plot_set

#加载语音信号
wave_path = SPEECH_AUDIO_DIR / "bird.wav"
waveform,sr = librosa.load(wave_path,sr=None,duration=5)

#设置参数调用librosa.stft
n_fft = 1024
hop_length = 512 #帧移
waveform_stft = librosa.stft(waveform,n_fft=n_fft,hop_length=hop_length)
#画图
plot_set.plot_waveform(waveform,sr)
plot_set.plot_waveform_fft(waveform,sr,n_fft=n_fft)
plot_set.plot_spectrogram(np.abs(waveform_stft))
