
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpeechEnhancementDataset
from model import (
    SimpleDenoisingCNN,
    DilatedDenoisingCNN,
    ResidualDenoisingCNN,
)
from train import split_by_speaker


def calculate_snr(clean, signal):
    """计算每条语音的参考波形SNR。"""

    clean = clean.astype(np.float64)
    signal = signal.astype(np.float64)

    error = signal - clean

    clean_power = np.sum(clean ** 2, axis=1)
    error_power = np.sum(error ** 2, axis=1)

    # 防止零功率造成数值错误
    eps = 1e-12

    snr = 10 * np.log10(
        np.maximum(clean_power, eps)
        / np.maximum(error_power, eps)
    )

    return snr


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    all_input_snr = []
    all_output_snr = []

    for mixture, clean in loader:
        predicted = model(mixture)

        mixture_np = mixture.numpy()
        clean_np = clean.numpy()
        predicted_np = predicted.numpy()

        input_snr = calculate_snr(
            clean_np, mixture_np
        )

        output_snr = calculate_snr(
            clean_np, predicted_np
        )

        all_input_snr.extend(input_snr.tolist())
        all_output_snr.extend(output_snr.tolist())

    input_snr = np.array(all_input_snr)
    output_snr = np.array(all_output_snr)

    improvement = output_snr - input_snr

    return input_snr, output_snr, improvement


def main():
    base_dir = Path(__file__).resolve().parent

    data_dir = (
        base_dir / "data" / "LibriSpeech" / "dev-clean"
    )

    output_dir = base_dir / "output"

    # 使用与训练一致的验证集划分
    _, val_files = split_by_speaker(data_dir)

    dataset = SpeechEnhancementDataset(
        data_dir=data_dir,
        files=val_files,
        max_files=10,
        duration=2.0,
        snr_db=5.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    model_configs = {
        "Original CNN": (
            SimpleDenoisingCNN,
            "best_model.pth",
        ),
        "Dilated CNN": (
            DilatedDenoisingCNN,
            "best_dilated_model.pth",
        ),
        "Residual CNN": (
            ResidualDenoisingCNN,
            "best_residual_model.pth",
        ),
    }

    for name, (model_class, filename) in model_configs.items():
        model = model_class()

        state_dict = torch.load(
            output_dir / filename,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict)

        input_snr, output_snr, improvement = (
            evaluate_model(model, loader)
        )

        print(f"\n===== {name} =====")
        print("样本数量:", len(improvement))

        print(
            f"平均输入SNR: {input_snr.mean():.2f} dB"
        )
        print(
            f"平均输出SNR: {output_snr.mean():.2f} dB"
        )
        print(
            f"平均SNR提升: {improvement.mean():+.2f} dB"
        )
        print(
            f"最小SNR提升: {improvement.min():+.2f} dB"
        )
        print(
            f"最大SNR提升: {improvement.max():+.2f} dB"
        )

        improved_count = np.sum(improvement > 0)

        print(
            f"改善样本数: "
            f"{improved_count}/{len(improvement)}"
        )

        print("逐条SNR提升:")
        for i, value in enumerate(improvement):
            print(f"  样本{i}: {value:+.2f} dB")


if __name__ == "__main__":
    main()
