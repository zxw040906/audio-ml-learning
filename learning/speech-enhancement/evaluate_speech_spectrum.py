from pathlib import Path

import numpy as np

from dataset import SpeechEnhancementDataset
from train import split_by_speaker
from analyze_spectrum import calculate_low_frequency_ratio


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "LibriSpeech" / "dev-clean"

    # 使用与上次模型评估相同的说话人划分
    _, val_files = split_by_speaker(data_dir)

    # 使用相同的 10 条验证语音
    dataset = SpeechEnhancementDataset(
        data_dir=data_dir,
        files=val_files,
        max_files=10,
        duration=2.0,
        snr_db=5,
    )

    ratios = []

    for index in range(len(dataset)):
        mixture, clean = dataset[index]

        # 将干净语音转换为 NumPy 数组
        clean_np = clean.numpy()

        # 计算该条语音的低频能量占比
        ratio = calculate_low_frequency_ratio(signal=clean_np,sample_rate=16_000,cutoff=500)

        # 保存计算结果
        ratios.append(ratio)

        print(f"样本 {index + 1}: {ratio:.2%}")

    # 计算全部验证语音的平均低频能量占比
    mean_ratio = np.average(ratios)

    print(f"\n平均低频能量占比: {mean_ratio:.2%}")


if __name__ == "__main__":
    main()