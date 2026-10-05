from pathlib import Path
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

import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

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