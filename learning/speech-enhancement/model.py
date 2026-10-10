import torch
import torch.nn as nn

class SimpleDenoisingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=9,
                padding=4,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=16,
                out_channels=16,
                kernel_size=9,
                padding=4,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=9,
                padding=4,
            ),
        )

    def forward(self, x):
        # 输入: [B, T]
        x = x.unsqueeze(1)

        # 卷积网络: [B, 1, T]
        x = self.network(x)

        # 输出: [B, T]
        return x.squeeze(1)


class DilatedDenoisingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                1, 16,
                kernel_size=9,
                padding=4,
                dilation=1,
            ),
            nn.ReLU(),

            nn.Conv1d(
                16, 16,
                kernel_size=9,
                padding=8,
                dilation=2,
            ),
            nn.ReLU(),

            nn.Conv1d(
                16, 16,
                kernel_size=9,
                padding=16,
                dilation=4,
            ),
            nn.ReLU(),

            nn.Conv1d(
                16, 16,
                kernel_size=9,
                padding=32,
                dilation=8,
            ),
            nn.ReLU(),

            nn.Conv1d(
                16, 1,
                kernel_size=9,
                padding=64,
                dilation=16,
            ),
        )

    def forward(self, x):
        # [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)

        x = self.network(x)

        # [B, 1, T] -> [B, T]
        return x.squeeze(1)


class ResidualDenoisingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(16, 16, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(16, 1, kernel_size=9, padding=4),
        )

        # 最后一层初始化为0
        # 使模型刚开始时预测的噪声为0
        last_conv = self.network[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        # x: [B, T]
        x_channels = x.unsqueeze(1)

        # 预测噪声
        predicted_noise = self.network(x_channels)
        predicted_noise = predicted_noise.squeeze(1)

        # 从输入中减去预测噪声
        predicted_clean = x - predicted_noise

        return predicted_clean



if __name__ == "__main__":
    model = SimpleDenoisingCNN()

    # 模拟一个Batch，4条音频，每条32000个采样点
    x = torch.randn(4, 32000)

    with torch.no_grad():
        y = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    print(
        "模型参数量:",
        sum(p.numel() for p in model.parameters())
    )

    dilated_model = DilatedDenoisingCNN()

    x = torch.randn(4, 32000)

    with torch.no_grad():
        y = dilated_model(x)

    print("空洞CNN输入:", x.shape)
    print("空洞CNN输出:", y.shape)

    print(
        "空洞CNN参数量:",
        sum(p.numel() for p in dilated_model.parameters())
    )

    residual_model = ResidualDenoisingCNN()
    
    x = torch.randn(4, 32000)

    with torch.no_grad():
        y = residual_model(x)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    print("初始最大误差:", torch.max(torch.abs(x - y)).item())

    print(
        "参数量:",
        sum(p.numel() for p in residual_model.parameters())
    )

