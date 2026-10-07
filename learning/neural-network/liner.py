# %% 导入依赖并读取图片
from pathlib import Path
from PIL import Image
from torch import nn
import torchvision
from IPython.display import display

#1.定位图片并转换为 Tensor
SCRIPT_DIR = Path(__file__).resolve().parent
img_path = SCRIPT_DIR / "assets" / "zxw.jpg"
with Image.open(img_path) as img:
    img = img.convert("RGB")

trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)
display(img)

# 定义线性网络
class Liner(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner = nn.Linear(in_features= 1076, out_features=720,)

    def forward(self,data):
        output = self.liner(data)
        return output

mynn = Liner()
input = img_tensor.unsqueeze(0)
output = mynn(input)
print(f"output shape is {output.shape}")
trans_PIL = torchvision.transforms.ToPILImage()
img_PIL = trans_PIL(output.squeeze(0))
display(img_PIL)

# %%
