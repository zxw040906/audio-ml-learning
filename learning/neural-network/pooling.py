from learning.common.paths import (
    output_dir,
)

from pathlib import Path
from PIL import Image
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


img_path = Path(__file__).resolve().parent / "assets" / "zxw.jpg"
img = Image.open(img_path).convert("RGB")
trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)

#构建池化的网络模型,继承nn.Module,包含MaxPool2d()
class MyPooling(nn.Module):
    def __init__(self):
        super().__init__()
        #最大池化
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        #均值池化
        self.pooling2 = nn.AvgPool2d(kernel_size=2,stride=2,ceil_mode=False)
    def forward(self, data):
        output = self.pooling2(data)
        return output

file = SummaryWriter(str(output_dir("neural-network", "log_network")))
#调用模型进行计算
mynn = MyPooling()
input_tensor = img_tensor.unsqueeze(0)
output = mynn(input_tensor)
print(output.shape)

file.add_image("Image",img_tensor)
file.add_image("Pooling",output.squeeze(0))

trans_PIL = torchvision.transforms.ToPILImage()
img_PIL = trans_PIL(output.squeeze(0))
img_PIL.show()
