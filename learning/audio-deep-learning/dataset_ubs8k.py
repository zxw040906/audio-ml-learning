from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _import_path in (str(_REPO_ROOT), str(_Path(__file__).resolve().parent)):
    if _import_path not in _sys.path:
        _sys.path.insert(0, _import_path)

from learning.common.paths import (
    URBANSOUND8K_ROOT,
)

import os
import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd
from plot_set import plot_waveform, plot_spectrogram


class UrbanSound8K(Dataset):
    def __init__(self, root_dir, excel_dir, transforms, target_samples, target_sr):
        super().__init__()
        self.root_dir = root_dir
        self.excel = pd.read_csv(excel_dir)
        self.transforms = transforms
        self.target_samples = target_samples
        self.target_sr = target_sr

    def reshape_if_necessary(self, waveform):
        if waveform.shape[1] > self.target_samples:
            waveform = waveform[:, 0:self.target_samples]
        elif waveform.shape[1] < self.target_samples:
            num_pad = self.target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, pad=(0, num_pad))
        return waveform

    def resample_if_necessary(self, waveform, sr):
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.target_sr
            )
        return waveform

    def mix_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem__(self, index):
        fold = f"fold{self.excel.iloc[index, 5]}"
        audio_name = f"{self.excel.iloc[index, 0]}"
        target = self.excel.iloc[index, 6]

        audio_path = os.path.join(self.root_dir, fold)
        wave_path = os.path.join(audio_path, audio_name)

        waveform, sr = torchaudio.load(wave_path)

        # 顺序很重要：先混单声道，再重采样，最后定长
        waveform = self.mix_if_necessary(waveform)
        waveform = self.resample_if_necessary(waveform, sr)
        waveform = self.reshape_if_necessary(waveform)

        waveform_mel = self.transforms(waveform)

        return waveform_mel, self.target_sr, target

    def __len__(self):
        return len(self.excel)


root_dir = str(URBANSOUND8K_ROOT / "audio")
excel_dir = str(URBANSOUND8K_ROOT / "metadata" / "UrbanSound8K.csv")

# n_fft 改大一点，n_mels 改小一点，避免 mel filterbank 警告
trans_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16_000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

mydata = UrbanSound8K(
    root_dir=root_dir,
    excel_dir=excel_dir,
    transforms=trans_mel,
    target_sr=16_000,
    target_samples=50_000
)

index = 5
waveform_mel, sr, target = mydata[index]

# waveform_mel 形状是 [1, n_mels, time]
# 去掉 channel 维度，变成 [n_mels, time] 再画图
plot_spectrogram(waveform_mel.squeeze(0))
print(target)
