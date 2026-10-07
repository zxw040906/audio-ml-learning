from pathlib import Path
from learning.common.paths import (
    output_dir,
)

import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

#加载数据集
img_path = str(Path(__file__).resolve().parent / "assets" / "nn.png")
img = Image.open(img_path)

#转换tensor类型
trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)

#文件中写入图像
file = SummaryWriter(str(output_dir("audio-deep-learning", "log_IMAGE")))
file.add_image(tag="Image",img_tensor=img_tensor)

#对图像归一化处理
trans_norm = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.3,0.6,0.1])
img_norm = trans_norm(img_tensor)

#文件中写入图像
file.add_image(tag="Norm",img_tensor=img_norm)

#更改尺寸
trans_resize = torchvision.transforms.Resize(size=(200,200))
img_resize = trans_resize(img_tensor)

#文件中写入图像
file.add_image(tag="Resize",img_tensor=img_resize)

#对图像进行任意裁剪
trans_crop = torchvision.transforms.RandomCrop(size=(50,60))
img_crop = trans_crop(img_tensor)

#文件中写入图像
file.add_image(tag="Crop",img_tensor=img_crop)

file.close()
