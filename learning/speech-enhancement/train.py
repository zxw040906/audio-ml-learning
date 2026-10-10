from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SpeechEnhancementDataset
from model import SimpleDenoisingCNN
from model import DilatedDenoisingCNN
from model import ResidualDenoisingCNN

def split_by_speaker(data_dir):
    """按说话人划分训练和验证文件。"""
    files = sorted(Path(data_dir).rglob("*.flac"))

    speaker_ids = sorted({
        path.parent.parent.name for path in files
    })

    if len(speaker_ids) < 2:
        raise ValueError("至少需要两个不同说话人")

    # 固定划分，保证可复现
    rng = np.random.default_rng(42)
    rng.shuffle(speaker_ids)

    n_val = max(1, int(len(speaker_ids) * 0.2))
    val_speakers = set(speaker_ids[:n_val])

    train_files = [
        p for p in files
        if p.parent.parent.name not in val_speakers
    ]
    val_files = [
        p for p in files
        if p.parent.parent.name in val_speakers
    ]

    return train_files, val_files


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for mixture, clean in loader:
        mixture = mixture.to(device)
        clean = clean.to(device)

        # 前向传播
        predicted = model(mixture)

        # 计算损失
        loss = criterion(predicted, clean)

        # 清空旧梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    for mixture, clean in loader:
        mixture = mixture.to(device)
        clean = clean.to(device)

        predicted = model(mixture)
        loss = criterion(predicted, clean)

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    torch.manual_seed(42)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "LibriSpeech" / "dev-clean"

    # 第一次只用少量样本，控制训练时间
    train_files, val_files = split_by_speaker(data_dir)

    train_dataset = SpeechEnhancementDataset(
        data_dir=data_dir,
        files=train_files,
        max_files=40,
        duration=2.0,
        snr_db=5.0,
    )

    val_dataset = SpeechEnhancementDataset(
        data_dir=data_dir,
        files=val_files,
        max_files=10,
        duration=2.0,
        snr_db=5.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # 第一次先用CPU，便于排查问题
    device = torch.device("cpu")

    model = ResidualDenoisingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 5
    best_val_loss = float("inf")

    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                output_dir / "best_residual_model.pth"
            )

    print("训练完成")
    print("最佳验证损失:", best_val_loss)


if __name__ == "__main__":
    main()
