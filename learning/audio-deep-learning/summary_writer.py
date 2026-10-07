from learning.common.paths import (
    output_dir,
)

from torch.utils.tensorboard import SummaryWriter
import torch
from dataset import mydata
file = SummaryWriter(str(output_dir("audio-deep-learning", "log_func")))
#y=2x
for x in range(20):
    file.add_scalar(tag="y=2x",scalar_value=2*x,global_step=x)
file.close()
#加载音频信号
file1 = SummaryWriter(str(output_dir("audio-deep-learning", "log_wav")))
flag = 0
for data in mydata:
    waveform,sr,name = data
    file1.add_audio(tag=f"{name}",snd_tensor=waveform,sample_rate=sr)
    flag+=1
    if flag == 8:
        exit()
file1.close()
