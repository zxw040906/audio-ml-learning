import torch
from torch import nn

# 1. 搭建模型
# 按照网络结构图修改下面的配置。输入张量的形状约定为：
# (批次大小, 通道数, 高度, 宽度)，例如 RGB 的 32×32 图片为 (N, 3, 32, 32)。
# ===== 通常只需要修改下面的配置 =====
INPUT_CHANNELS = 3
INPUT_HEIGHT = 32
INPUT_WIDTH = 32

# 按顺序配置每个卷积层：
# out_channels：卷积层输出通道数（特征图数量）
# kernel：卷积核大小；这里使用正方形卷积核
# stride：卷积步长
# output_size：(卷积后的高度, 卷积后的宽度)，注意这是池化之前的尺寸
CONV_LAYERS = [
    {"out_channels": 32, "kernel": 5, "stride": 1, "output_size": (32, 32)},
    {"out_channels": 32, "kernel": 5, "stride": 1, "output_size": (16, 16)},
    {"out_channels": 64, "kernel": 5, "stride": 1, "output_size": (8, 8)},
]

# 池化配置：键是卷积层编号（从 1 开始），值为 (池化核大小, 池化步长)。
# 例如 {1: (2, 2)} 表示第一层卷积后接 2×2 最大池化，步长为 2。
# 某层卷积后不需要池化时，不要在字典中添加该层编号。
POOLS = {1: (2, 2), 2: (2, 2), 3: (2, 2)}
HIDDEN_UNITS = 64  # 全连接隐藏层的神经元数量
OUTPUT_UNITS = 10  # 输出维度；分类时通常等于类别数
# ==============================================


def calc_padding(before, after, kernel, stride):
    """根据卷积输入/输出尺寸反推对称 padding。"""
    # PyTorch 卷积尺寸公式：输出 = floor((输入 + 2*padding - 卷积核) / 步长) + 1。
    # 这里要求高度、宽度分别能用非负整数的对称 padding 精确得到目标尺寸。
    numerator = (after - 1) * stride - before + kernel
    if numerator < 0 or numerator % 2:
        raise ValueError("该输入/输出尺寸无法用非负整数对称 padding 实现")
    return numerator // 2


class MyNetwork(nn.Module):
    """根据上面的配置依次构造卷积、池化和全连接层。"""
    def __init__(self):
        super().__init__()
        layers = []
        # 记录当前特征图的通道数和空间尺寸，供下一层计算输入参数及 padding 使用。
        channels = INPUT_CHANNELS
        height, width = INPUT_HEIGHT, INPUT_WIDTH

        for i, config in enumerate(CONV_LAYERS, start=1):
            out_h, out_w = config["output_size"]
            # 分别按高度和宽度计算 padding；Conv2d 的 padding 参数顺序为 (高, 宽)。
            pad_h = calc_padding(height, out_h, config["kernel"], config["stride"])
            pad_w = calc_padding(width, out_w, config["kernel"], config["stride"])
            layers.append(nn.Conv2d(
                channels, config["out_channels"], config["kernel"],
                config["stride"], padding=(pad_h, pad_w)
            ))
            channels = config["out_channels"]
            height, width = out_h, out_w

            # 如果当前卷积层编号出现在 POOLS 中，就在卷积后添加最大池化层。
            if i in POOLS:
                pool_kernel, pool_stride = POOLS[i]
                layers.append(nn.MaxPool2d(pool_kernel, pool_stride))
                # 更新池化后的尺寸，供后续卷积层或尺寸推导使用。
                height = (height - pool_kernel) // pool_stride + 1
                width = (width - pool_kernel) // pool_stride + 1

        # Sequential 按顺序执行所有卷积和池化层。
        self.features = nn.Sequential(*layers)
        # 将 (批次, 通道, 高, 宽) 展开为 (批次, 特征数)，以便输入全连接层。
        self.flatten = nn.Flatten()
        # 用一个虚拟输入自动推导展平后的特征数，避免手工计算全连接层输入维度。
        with torch.no_grad():
            sample = torch.zeros(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
            flattened_size = self.flatten(self.features(sample)).shape[1]
        # 全连接部分：展平特征 -> 隐藏层 -> 输出层。
        # 如需常见的非线性网络，可在两个 Linear 之间加入 nn.ReLU()。
        self.model = nn.Sequential(
            nn.Linear(flattened_size, HIDDEN_UNITS),
            nn.Linear(HIDDEN_UNITS, OUTPUT_UNITS),
        )

    def forward(self, data):
        # 前向传播顺序：卷积/池化提取特征 -> 展平 -> 全连接层得到输出。
        output = self.features(data)
        output = self.flatten(output)
        output = self.model(output)
        return output

# 2. 准备示例数据（实际训练时替换为数据集和 DataLoader）
# 这里用回归任务演示：10 个输入样本，每个样本对应 10 维浮点目标。
# batch_size=2 表示每次取 2 个样本；最后一批不足 2 个时也照常训练。
data = torch.ones(size=(10, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH))
target = torch.ones(size=(10, OUTPUT_UNITS), dtype=torch.float32)
batch_size = 2

# 3. 创建模型、损失函数和优化器。优化器应在训练循环外创建一次。
learning_rate = 1e-5
num_epochs = 50
mynn = MyNetwork()
loss_func = nn.MSELoss(reduction='mean')
optimiser = torch.optim.RMSprop(params=mynn.parameters(), lr=learning_rate)

# 小批量训练示意：外层遍历 epoch，内层按 batch_size 切分数据。
# 每个 epoch 都会重新遍历全部样本，但每次参数更新只使用当前批次。
for epoch in range(num_epochs):
    for start in range(0, data.shape[0], batch_size):
        # 取出当前批次的输入和对应目标，切片会自动处理最后一个较小批次。
        batch_data = data[start:start + batch_size]
        batch_target = target[start:start + batch_size]

        # 3. 前向计算并计算当前批次的损失。
        y_pred = mynn(batch_data)
        loss = loss_func(y_pred, batch_target)

        # 4. 清梯度、反向传播、更新参数。
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, 最后一批 loss={loss.item():.6f}")

print(f"最后一批预测形状: {y_pred.shape}")
