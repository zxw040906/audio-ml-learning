from learning.common.paths import (
    SPEECH_AUDIO_DIR,
)

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

wave_path = SPEECH_AUDIO_DIR / "bird.wav"
waveform, sample_rate = librosa.load(wave_path, sr=None)   # sr=None：保持文件原始采样率

def Calc_RMS(waveform, frame_length, hop_length):
    """
    计算语音信号的能量包络（RMS，Root Mean Square）。

    参数
    ----
    waveform     : np.ndarray  一维数组，语音信号
    frame_length : int         帧长
    hop_length   : int         帧移

    返回
    ----
    RMS          : np.ndarray  一维数组，能量包络
    """
    # 1. 分帧
    frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length).T  # 转置为 (frame_num, frame_length)
    # 2. 计算每一帧的 RMS 值
    RMS = np.sqrt(np.mean(frames**2, axis=1))
    return RMS
#3. 设置分帧参数
frame_length = 1024   # 帧长：每帧 1024 个点；48kHz 下约 1024/48000 ≈ 21.3 ms
hop_length = int(frame_length * 0.5)      # 帧移：每次前进 512 个点
#4. 计算能量包络
RMS = Calc_RMS(waveform, frame_length, hop_length)
#5.利用librosa的rms函数计算能量包络
rms_librosa = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
#6.利用Calc_RMS画出图像与利用librosa的rms函数画出图像然后对比二者图像差距
plt.figure(figsize=(10, 4))
plt.plot(RMS, label='Custom RMS')
plt.plot(rms_librosa, label='Librosa RMS')
plt.xlabel('Frame')
plt.ylabel('RMS')
plt.title('RMS Envelope Comparison')
plt.legend()
plt.show()
