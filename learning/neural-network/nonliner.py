
# %% 导入依赖并读取图片
from pathlib import Path
from PIL import Image
from torch import nn
import torchvision
from IPython.display import display

#1.定位图片并转换为 Tensor
"""
图片路径最好不要写成相对终端目录的字符串，
例如 `"assets/image.jpg"`。这种写法会随着你从哪个目录启动程序而改变含义。
在你的脚本中复制下面的代码，用 `Path(__file__).resolve().parent` 先取得当前 `.py` 文件所在目录，再拼接图片相对位置：
这适合图片跟某个脚本/项目一起保存的情况；即使从仓库根目录、别的工作目录或用脚本绝对路径启动，都能找到同一张图片。
`Path` 的 `/` 运算符会按当前操作系统拼接路径，不需要手动写 `/` 或 `\\`。
"""
SCRIPT_DIR = Path(__file__).resolve().parent
img_path = SCRIPT_DIR / "assets" / "image.jpg"
with Image.open(img_path) as img:
    img = img.convert("RGB")

trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)
display(img)
"""
ReLu函数:(一般用于二分类问题)
优点:解决梯度消失问题(尤其导数为非0值),并且计算速度快(只输出0,1)
缺点:没有输出负数,不是zero-centered ,可能会出现Dead ReLu Problem
即一些神经元未激活到,因为计算速率太高导致训练过程中参数更新太快

Tanh函数:
优点:输出正负,解决了zero-centered问题
缺点:梯度消失问题以及幂运算问题仍旧存在

Sigmoid函数:(一般用于多分类问题):
优点:适用于多分类问题,将输出归一化
缺点:可能产生梯度消失,且权重更新过程不是收敛过程,Sigmoid函数的输出值恒大于0
易导致模型训练收敛速度变慢

"""
#2.构建 ReLU 并处理图片
class NonLiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLu = nn.ReLU()
        self.Tanh = nn.Tanh() 
        self.Sigmoid = nn.Sigmoid()
    def forward(self,data):
        output = self.Sigmoid(data)
        return output

#3.调用模型进行计算
mynn = NonLiner()
input = img_tensor.unsqueeze(0)
output = mynn(input)
print(output.shape)

#4.将处理结果转回图片并在 VS Code 中预览
trans_PIL = torchvision.transforms.ToPILImage()
img_PIL = trans_PIL(output.squeeze(0))
display(img_PIL)

print(f"bias={(img_tensor - output.squeeze(0)).sum()}")

# %%
