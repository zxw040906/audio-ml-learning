from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = next(
    parent for parent in _Path(__file__).resolve().parents
    if (parent / "learning" / "common" / "paths.py").is_file()
)
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from learning.common.paths import (
    REPO_ROOT, URBANSOUND8K_ROOT, YESNO_AUDIO_DIR, SPEECH_AUDIO_DIR, output_dir
)

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
#1. 读取音频文件
bird_path = SPEECH_AUDIO_DIR / "bird.wav"
bird, sr = librosa.load(bird_path, sr=None ,duration=5)   # sr=None：保持文件原始采样率

time_path = SPEECH_AUDIO_DIR / "time.wav"
time, sr = librosa.load(time_path, sr=None ,duration=5)   # sr=None：保持文件原始采样率

water_path = SPEECH_AUDIO_DIR / "water.wav"
water, sr = librosa.load(water_path, sr=None ,duration=5)   # sr=None：保持文件原始采样率
#2.获取Spectral Centroid
sc_bird = librosa.feature.spectral_centroid(y=bird, sr=sr , n_fft=1024)[0]
sc_time = librosa.feature.spectral_centroid(y=time, sr=sr , n_fft=1024)[0]
sc_water = librosa.feature.spectral_centroid(y=water, sr=sr , n_fft=1024)[0]
#3.画出Spectral Centroid图像
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveshow(bird, sr=sr, ax=ax[0])
ax[0].label_outer()
ax[0].set(title='Bird', xlabel=None, ylabel='Amplitude')
librosa.display.waveshow(time, sr=sr, ax=ax[1])
ax[1].label_outer()
ax[1].set(title='Time', xlabel=None, ylabel='Amplitude')
librosa.display.waveshow(water, sr=sr, ax=ax[2])
ax[2].set(title='Water', xlabel='Time (s)', ylabel='Amplitude')
plt.show()