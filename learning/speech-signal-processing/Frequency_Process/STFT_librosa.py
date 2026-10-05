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

import os
import sys

import librosa
from matplotlib import pyplot as plt
import numpy as np

#把脚本所在目录加入模块搜索路径,保证从任意工作目录运行都能导入plot_set
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import plot_set

#计算工程根目录(脚本所在目录的上一级),音频文件都放在根目录下
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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