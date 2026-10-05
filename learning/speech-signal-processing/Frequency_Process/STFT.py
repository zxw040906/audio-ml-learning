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

from enum import auto

import librosa
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载音频文件
time_path = str(SPEECH_AUDIO_DIR / "bird.wav")
waveform, sr = librosa.load(time_path, sr=None,duration=5)  # sr=None 保持原始采样率

# 2. 信号分帧: 补零 -> 分帧 -> 加窗
"""frame_size 帧长, overlap 重叠, hop_size 帧移 = 帧长 - 重叠"""
frame_size = 1024
hop_size = int(frame_size * 0.5)   # 50% 重叠

# 补零:让 (N - frame) 能被 hop 整除
"""
令 r = (N - frame) % hop        # 当前的"余数",也就是最后一帧差多少个点才凑满
   r = 0  → 已经整除,不用补
   r != 0 → 差 hop - r 个点,补上就整了

"""
r = (len(waveform) - frame_size) % hop_size
waveform = np.pad(waveform, (0, (hop_size - r) % hop_size), mode='constant')

# 分帧:窗口数 = (补零后长度 - 帧长)//帧移 + 1
frame_num = (len(waveform) - frame_size) // hop_size + 1
"""frames = np.stack([waveform[t*hop_size: t*hop_size+frame_size] for t in range(frame_num)])"""
# 生成索引矩阵,每一行是一个帧的索引
row = np.tile(np.arange(0, frame_size), (frame_num, 1))
# 生成每一帧的起始索引
column = np.tile(np.arange(0,frame_num*hop_size, hop_size), (frame_size, 1)).T
# 通过索引矩阵获取每一帧的信号
idx = row + column
waveform_frames = waveform[idx]
# 3. 加窗
waveform_frames *= np.hanning(frame_size)  # 汉宁窗

#将信号进行傅里叶变换
n_fft = 1024  # FFT点数
waveform_stft = np.fft.rfft(waveform_frames, n_fft)  # 对每一帧进行傅里叶变换
"""
注意:功率量转dB用 10*lg( ),幅度量转dB用 20*lg( )。
因为 P ∝ A^2,所以 10*log10(P) = 20*log10(A)。
此处 waveform_pow 是功率量，故用 10*log10。
"""
waveform_pow = np.abs(waveform_stft)**2/n_fft 
waveform_db = 10 * np.log10(waveform_pow)  # 转换为分贝
#绘制波形
plt.figure(figsize=(10,10))
plt.imshow(waveform_db,aspect="auto",origin='lower')
#此处y_ticks为y轴(频率)刻度,因为只做了实数部分所以需要取一半采样点
y_ticks = np.arange(0,n_fft//2+1,50)
plt.yticks(ticks=y_ticks,labels=[str(round(t * sr / n_fft, 1)) for t in y_ticks])
plt.title("Waveform_STFT")
plt.show()
