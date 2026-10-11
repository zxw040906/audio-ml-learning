from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpeechEnhancementDataset
from model import ResidualDenoisingCNN
from train import split_by_speaker
from evaluate_batch import calculate_snr, evaluate_model


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "LibriSpeech" / "dev-clean"
    model_path = base_dir / "output" / "best_residual_model.pth"

    # 1. 与训练阶段使用相同的说话人划分
    _, val_files = split_by_speaker(data_dir)

    # 2. 加载固定的残差模型
    model = ResidualDenoisingCNN()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # 3. 只改变测试输入的 SNR
    snr_levels = [0, 5, 10, 15]

    for snr_db in snr_levels:
        dataset = SpeechEnhancementDataset(
            data_dir=data_dir,
            files=val_files,
            max_files=10,
            duration=2.0,
            snr_db=snr_db,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        # 4. 计算每条语音的输入、输出 SNR
        input_snr, output_snr, improvement = evaluate_model(
            model, loader
        )

        print(f"\n===== 输入目标 SNR: {snr_db} dB =====")
        print(f"样本数量: {len(improvement)}")
        print(f"平均输入 SNR: {input_snr.mean():.2f} dB")
        print(f"平均输出 SNR: {output_snr.mean():.2f} dB")
        print(f"平均提升: {improvement.mean():+.2f} dB")
        print(f"最小提升: {improvement.min():+.2f} dB")
        print(f"最大提升: {improvement.max():+.2f} dB")
        print(
            f"改善样本数: "
            f"{np.sum(improvement > 0)}/{len(improvement)}"
        )


if __name__ == "__main__":
    main()