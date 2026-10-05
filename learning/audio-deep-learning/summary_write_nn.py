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

from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
#参照nn.png来写神经网络
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
        self.pooling3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.line1 = nn.Linear(in_features=64*4*4,out_features=64)
        self.line2 = nn.Linear(in_features=64,out_features=10)
    def forward(self,data):
        output = self.conv1(data)
        output = self.pooling1(output)
        output = self.conv2(output)
        output = self.pooling2(output)
        output = self.conv3(output)
        output = self.pooling3(output)
        output = self.flatten(output)
        output = self.line1(output)
        output = self.line2(output)
        return output

mynn = MyNet()
input = torch.Tensor(1,3,32,32)
output = mynn(input)
print(output.shape)

#做出神经网络图,用于论文极为友好
file = SummaryWriter(str(output_dir("audio-deep-learning", "log_net")))
file.add_graph(model=mynn,input_to_model=input)
file.close()