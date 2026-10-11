from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _import_path in (str(_REPO_ROOT), str(_Path(__file__).resolve().parent)):
    if _import_path not in _sys.path:
        _sys.path.insert(0, _import_path)

from pathlib import Path
from learning.common.paths import (
    output_dir,
)

import torch
import torch.nn.functional
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Module 
from torch.utils.tensorboard import SummaryWriter
"""
padding计算:
padding = (after_dim - 1) * stride + kernel_size - before_dim
"""
#此处为二维卷积的图像验证测试
img_path = str(Path(__file__).resolve().parent / "assets" / "image.jpg")
img = Image.open(img_path)
trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)
#此处因为输入是4维的,所以需要升高维度为4维
img_tensor = img_tensor.unsqueeze(0)

class MyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=2,stride=1,padding=1)
    def forward(self,data):
        output = self.conv(data)
        return output

mynn = MyConv()
#创建文件 
file = SummaryWriter(str(output_dir("neural-network", "log_conv")))

output = mynn(img_tensor)
print(output.shape)

trans_PIL = torchvision.transforms.ToPILImage()
img_PIL = trans_PIL(output.squeeze(0))
file.add_graph(mynn,img_tensor)
file.close()
img_PIL.show()
