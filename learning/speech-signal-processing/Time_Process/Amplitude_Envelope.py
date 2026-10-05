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
import matplotlib.pyplot as plt

# 读取音频文件
music_path = str(SPEECH_AUDIO_DIR / "time.wav")
waveform , sr = librosa.load(music_path, sr=None,duration=5)  # sr=None：保持文件原始采样率


# 计算音频信号幅值包络(分帧计算手写)
def Calc_Amplitude_Envelope(waveform,frame_length,hop_length):
    #分帧操作
    if(len(waveform)-frame_length)%hop_length!=0:
        frame_num = int((len(waveform)-frame_length)/hop_length) + 1
        #补零个数
        pad_num = frame_num*hop_length+frame_length-len(waveform)
        waveform = np.pad(waveform,(0,int(pad_num)),'wrap')
    frame_num = int(np.ceil((len(waveform)-frame_length)/hop_length)) + 1
    waveform_ae = []
    for t in range(frame_num):
        current_frame = waveform[t*hop_length:t*hop_length+frame_length]
        current_ae = max(current_frame)
        waveform_ae.append(current_ae)
    return np.array(waveform_ae)
#librosa自带的分帧计算方法
def Calc_Amplitude_Envelope_librosa(waveform,frame_length,hop_length):
    frame_num = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length)
    waveform_ae = np.max(frame_num, axis=0)
    return waveform_ae
#设置参数
frame_size = 1024
hop_size = int(frame_size*0.5)
waveform_ae = Calc_Amplitude_Envelope(waveform=waveform,frame_length=frame_size,hop_length=hop_size)
librosa_ae = Calc_Amplitude_Envelope_librosa(waveform=waveform,frame_length=frame_size,hop_length=hop_size)
#画出两者幅值图进行比较
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(waveform, color='gray', alpha=0.5, label='Waveform')
plt.plot(np.arange(len(waveform_ae)) * hop_size, waveform_ae, color='red', label='Amplitude Envelope (Custom)')
plt.title('Amplitude Envelope (Custom)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(waveform, color='gray', alpha=0.5, label='Waveform')
plt.plot(np.arange(len(librosa_ae)) * hop_size, librosa_ae, color='blue', label='Amplitude Envelope (Librosa)')
plt.title('Amplitude Envelope (Librosa)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()