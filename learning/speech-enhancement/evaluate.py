from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dataset import SpeechEnhancementDataset
from model import SimpleDenoisingCNN
from model import DilatedDenoisingCNN
from model import ResidualDenoisingCNN
from train import split_by_speaker


def calculate_snr(clean, signal):
    """计算信号相对于干净语音的SNR。"""
    clean = np.asarray(clean, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    error = signal - clean

    signal_power = np.sum(clean ** 2)
    error_power = np.sum(error ** 2)

    return 10 * np.log10(
        signal_power / max(error_power, 1e-12)
    )


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "LibriSpeech" / "dev-clean"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. 获取与训练一致的验证文件
    _, val_files = split_by_speaker(data_dir)

    dataset = SpeechEnhancementDataset(
        data_dir=data_dir,
        files=val_files,
        max_files=10,
        duration=2.0,
        snr_db=5.0,
    )

    # 2. 加载模型
    model = ResidualDenoisingCNN()

    model_path = output_dir / "best_residual_model.pth"

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # 3. 获取一条验证样本
    mixture, clean = dataset[0]

    # 4. 模型推理
    with torch.no_grad():
        predicted = model(
            mixture.unsqueeze(0)
        ).squeeze(0)

    # 5. 转为NumPy
    mixture_np = mixture.numpy()
    clean_np = clean.numpy()
    predicted_np = predicted.numpy()

    # 6. 计算SNR
    input_snr = calculate_snr(
        clean_np, mixture_np
    )

    output_snr = calculate_snr(
        clean_np, predicted_np
    )

    improvement = output_snr - input_snr

    print(f"输入SNR: {input_snr:.2f} dB")
    print(f"输出SNR: {output_snr:.2f} dB")
    print(f"SNR提升: {improvement:+.2f} dB")

    # 7. 统一缩放，防止保存音频时削波
    peak = max(
        np.max(np.abs(clean_np)),
        np.max(np.abs(mixture_np)),
        np.max(np.abs(predicted_np)),
    )

    gain = min(1.0, 0.99 / max(peak, 1e-12))

    sr = 16000

    sf.write(
        output_dir / "residual_clean.wav",
        clean_np * gain,
        sr,
        subtype="FLOAT",
    )

    sf.write(
        output_dir / "residual_mixture.wav",
        mixture_np * gain,
        sr,
        subtype="FLOAT",
    )

    sf.write(
        output_dir / "residual_predicted.wav",
        predicted_np * gain,
        sr,
        subtype="FLOAT",
    )

    print("试听音频已保存到:", output_dir)


if __name__ == "__main__":
    main()
