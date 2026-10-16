
from pathlib import Path

import numpy as np
import torch

from dataset import SpeechEnhancementDataset
from model import ResidualDenoisingCNN
from train import split_by_speaker
from audio_utils import calculate_band_and_total_mse, calculate_mse


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "LibriSpeech" / "dev-clean"
    model_path = base_dir / "output" / "best_residual_model.pth"

    _, val_files = split_by_speaker(data_dir)

    model = ResidualDenoisingCNN()
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    sample_rate = 16000
    bands = [
        (0, 500),
        (500, 1000),
        (1000, 3000),
        (3000, 8000),
    ]

    # 只使用 dataset 中的干净语音，不使用 mixture
    dataset = SpeechEnhancementDataset(
        data_dir=data_dir,
        files=val_files,
        max_files=10,
        duration=2.0,
        snr_db=5.0,
        noise_type="white",
    )

    band_results = {band: [] for band in bands}
    total_mse_values = []

    for index in range(len(dataset)):
        mixture, clean = dataset[index]

        # 练习 1：将干净语音送入模型
        with torch.no_grad():
            predicted = model(
                clean.unsqueeze(0)
            ).squeeze(0)

        clean_np = clean.numpy()
        predicted_np = predicted.squeeze(0).numpy()

        # 练习 2：计算干净输入经过模型后的整体误差
        total_mse = calculate_mse(predicted_np, clean_np)
        total_mse_values.append(total_mse)

        for low_freq, high_freq in bands:
            band_mse, _ = calculate_band_and_total_mse(
                # 练习 3：指定预测信号和参考信号
                predicted=predicted_np,
                clean=clean_np,
                sample_rate=sample_rate,
                low_freq=low_freq,
                high_freq=high_freq,
            )

            band_results[(low_freq, high_freq)].append(band_mse)

    print("\n干净语音输入实验")
    print(f"样本数量: {len(dataset)}")
    print(f"{'频段':<18}{'输出误差 MSE':>18}")
    print("-" * 38)

    band_sum = 0.0

    for low_freq, high_freq in bands:
        mean_band_mse = np.mean(
            band_results[(low_freq, high_freq)]
        )
        band_sum += mean_band_mse

        print(
            f"{low_freq}-{high_freq} Hz".ljust(18)
            + f"{mean_band_mse:>18.6f}"
        )

    mean_total_mse = np.mean(total_mse_values)

    print("-" * 38)
    print(f"{'整体':<18}{mean_total_mse:>18.6f}")
    print(
        "四频段误差检查:",
        np.allclose(band_sum, mean_total_mse),
    )


if __name__ == "__main__":
    main()
