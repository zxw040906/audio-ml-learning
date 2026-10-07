"""按网络结构图填写参数，自动推算全连接层输入并检查输出形状。"""

import torch
from torch import nn


# %% 1. 参数区：按结构图修改这里
# 输入格式为 [batch, channel, height, width]。
INPUT_CHANNELS = 3
INPUT_HEIGHT = 32
INPUT_WIDTH = 32

# 每个字典描述一层卷积：输出特征图数、卷积核大小、步长、目标输出高宽。
# padding 不用填写，程序会按 conv2.py 中的公式反算。
# 图中三层卷积的输出通道数分别为 32、32、64；空间尺寸为 32、16、8。
CONV_LAYERS = [
    {"out_channels": 32, "kernel_size": 5, "stride": 1, "output_size": (32, 32)},
    {"out_channels": 32, "kernel_size": 5, "stride": 1, "output_size": (16, 16)},
    {"out_channels": 64, "kernel_size": 5, "stride": 1, "output_size": (8, 8)},
]

# 字典的键表示“第几层卷积之后做池化”，层号从 1 开始。
# 图中每一层卷积后都接 2×2 最大池化，步长为 2。
POOL_AFTER_CONV = {
    1: {"kernel_size": 2, "stride": 2},
    2: {"kernel_size": 2, "stride": 2},
    3: {"kernel_size": 2, "stride": 2},
}

HIDDEN_UNITS = 64
OUTPUT_UNITS = 10


def calculate_padding(input_size, kernel_size, stride, output_size):
    """按 conv2.py 的尺寸公式反算 padding，返回 (pad_h, pad_w)。"""
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    input_h, input_w = input_size
    output_h, output_w = output_size

    # after = (before - kernel + 2 * padding) / stride + 1
    # 移项得到 padding = ((after - 1) * stride - before + kernel) / 2。
    numerator_h = (output_h - 1) * stride_h - input_h + kernel_h
    numerator_w = (output_w - 1) * stride_w - input_w + kernel_w
    if numerator_h % 2 or numerator_w % 2:
        raise ValueError(
            "目标输出尺寸无法用对称整数 padding 实现："
            f"input={input_size}, kernel={kernel_size}, stride={stride}, output={output_size}"
        )
    padding = numerator_h // 2, numerator_w // 2
    if padding[0] < 0 or padding[1] < 0:
        raise ValueError(
            f"计算出的 padding 不能为负数：input={input_size}, kernel={kernel_size}, "
            f"stride={stride}, output={output_size}, padding={padding}"
        )
    return padding


#2. 按参数自动构造网络
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        feature_layers = []
        in_channels = INPUT_CHANNELS

        current_height, current_width = INPUT_HEIGHT, INPUT_WIDTH
        for layer_number, config in enumerate(CONV_LAYERS, start=1):
            out_channels = config["out_channels"]
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            output_size = config["output_size"]
            padding = calculate_padding(
                (current_height, current_width),
                kernel_size,
                stride,
                output_size,
            )
            feature_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

            pool_config = POOL_AFTER_CONV.get(layer_number)
            if pool_config is not None:
                feature_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_config["kernel_size"],
                        stride=pool_config["stride"],
                    )
                )
                pool_kernel = pool_config["kernel_size"]
                pool_stride = pool_config.get("stride", pool_kernel)
                if isinstance(pool_kernel, int):
                    pool_kernel_h = pool_kernel_w = pool_kernel
                else:
                    pool_kernel_h, pool_kernel_w = pool_kernel
                if isinstance(pool_stride, int):
                    pool_stride_h = pool_stride_w = pool_stride
                else:
                    pool_stride_h, pool_stride_w = pool_stride
                current_height = (output_size[0] - pool_kernel_h) // pool_stride_h + 1
                current_width = (output_size[1] - pool_kernel_w) // pool_stride_w + 1
            else:
                current_height, current_width = output_size

            # 下一层卷积的输入通道数等于本层输出通道数。
            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)
        self.flatten = nn.Flatten()

        # 用虚拟输入计算 Flatten 后的长度，免去手工填写 Linear.in_features。
        with torch.no_grad():
            example = torch.zeros(
                1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH
            )
            flattened_features = self.flatten(self.features(example)).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_features, HIDDEN_UNITS),
            nn.Linear(HIDDEN_UNITS, OUTPUT_UNITS),
        )

    def forward(self, data):
        data = self.features(data)
        data = self.flatten(data)
        return self.classifier(data)


#3. 验证模板参数并打印各阶段的 Tensor 形状

model = MyNetwork()
sample = torch.zeros(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

with torch.no_grad():
    feature_output = model.features(sample)
    flattened = model.flatten(feature_output)
    output = model(sample)

print("输入形状：", tuple(sample.shape))
print("卷积/池化后：", tuple(feature_output.shape))
print("Flatten 后：", tuple(flattened.shape))
print("网络输出：", tuple(output.shape))
print("各卷积层自动计算的 padding：")
height, width = INPUT_HEIGHT, INPUT_WIDTH
for layer_number, config in enumerate(CONV_LAYERS, start=1):
    pad = calculate_padding(
        (height, width), config["kernel_size"], config["stride"], config["output_size"]
    )
    print(f"  {pad}")
    height, width = config["output_size"]
    # 池化后尺寸作为下一卷积层输入尺寸。
    pool = POOL_AFTER_CONV.get(layer_number)
    if pool:
        kernel = pool["kernel_size"]
        stride = pool.get("stride", kernel)
        height = (height - kernel) // stride + 1
        width = (width - kernel) // stride + 1
# %%