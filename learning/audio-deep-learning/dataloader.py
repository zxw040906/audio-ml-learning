from learning.common.paths import (
    YESNO_AUDIO_DIR,
)

from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch
from pathlib import Path

#定义数据集类
class MyData(Dataset):
    def __init__(self,root_dir,label_dir,target_samples,target_sr):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = Path(self.root_dir) / self.label_dir
        self.audio_file = sorted(self.path.iterdir())
        self.target_samples = target_samples
        self.target_sr = target_sr
    def reshape_if_necessary(self,waveform):
        if waveform.shape[1]>self.target_samples:
            waveform = waveform[:,0:self.target_samples]
        elif waveform.shape[1]<self.target_samples:
            pad_num = self.target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform,pad=(0,pad_num))
        return waveform
    def resample_if_necessary(self,wavefom,sr):
        if sr != self.target_sr:
            wavefom = torchaudio.functional.resample(wavefom,orig_freq=sr,new_freq=self.target_sr)
        return wavefom
    def mix_if_necessary(self,waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform,dim=0,keepdim=True)
        return waveform
    def __getitem__(self, index):
        audio_path = self.audio_file[index]
        audio,sr = torchaudio.load(audio_path)
        audio = self.mix_if_necessary(audio)
        audio = self.resample_if_necessary(audio,sr)
        audio = self.reshape_if_necessary(audio)
        return audio, audio_path.name
    def __len__(self):
        return len(self.audio_file)

#设定batch_size大小,加载数据集
root_dir = str(YESNO_AUDIO_DIR.parent)
label_dir = YESNO_AUDIO_DIR.name
target_samples = 20_000
target_sr = 8_000
mydataset = MyData(root_dir=root_dir,label_dir=label_dir,target_samples=target_samples,target_sr=target_sr)
data_batch = DataLoader(dataset = mydataset,batch_size = 10,shuffle = True,drop_last = False)
interation = 0
for data in data_batch:
    waveform,name = data 
    interation += 1
    print(f"the{interation}th and name is {name}")
