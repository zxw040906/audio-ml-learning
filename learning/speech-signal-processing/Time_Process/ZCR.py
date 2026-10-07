from learning.common.paths import (
    SPEECH_AUDIO_DIR,
)

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

#1.加载信号
wave_path = SPEECH_AUDIO_DIR / "bird.wav"
waveform, sample_rate = librosa.load(wave_path, sr=None)   # sr=None：保持文件原始采样率
#2.定义计算过零率的函数
def Calc_ZCR(waveform, frame_length, hop_length):
    """
    计算语音信号的过零率（ZCR，Zero Crossing Rate）。

    参数
    ----
    waveform     : np.ndarray  一维数组，语音信号
    frame_length : int         帧长
    hop_length   : int         帧移

    返回
    ----
    ZCR          : np.ndarray  一维数组，过零率
    """
    # 1. 分帧
    frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length).T  # 转置为 (frame_num, frame_length)
    # 2. 计算每一帧的过零率
    ZCR = np.sum(np.abs(np.diff(np.sign(frames))), axis=1) / (2 * frame_length)
    return ZCR
#3. 设置分帧参数
frame_length = 1024   # 帧长：每帧 1024 个点；48kHz 下约 1024/48000 ≈ 21.3 ms
hop_length = int(frame_length * 0.5)      # 帧移：每次前进 512 个点
#4. 计算过零率
ZCR = Calc_ZCR(waveform, frame_length, hop_length)
#5.利用librosa的zcr函数计算过零率
zcr_librosa = librosa.feature.zero_crossing_rate(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
#6.利用Calc_ZCR画出图像与利用librosa的zcr函数画出图像然后对比二者图像差距
plt.figure(figsize=(10, 4))
plt.plot(ZCR, label='Custom ZCR')
plt.plot(zcr_librosa, label='Librosa ZCR')
plt.xlabel('Frame')
plt.ylabel('ZCR')
plt.title('ZCR Comparison')
plt.legend()
plt.show()
