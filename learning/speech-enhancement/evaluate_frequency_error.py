
from pathlib import Path

import numpy as np
import torch

from dataset import SpeechEnhancementDataset
from model import ResidualDenoisingCNN
from train import split_by_speaker
from audio_utils import (
    calculate_mse,
    calculate_band_and_total_mse,
    calculate_band_correlation_from_energies,
)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "LibriSpeech" / "dev-clean"
    model_path = base_dir / "output" / "best_residual_model.pth"

    # 使用与训练阶段相同的说话人划分
    _, val_files = split_by_speaker(data_dir)

    # 加载固定的残差降噪模型
    model = ResidualDenoisingCNN()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # 实验配置
    noise_types = ["white", "low_frequency"]

    bands = [
        (0, 500),
        (500, 1000),
        (1000, 3000),
        (3000, 8000),
    ]

    files_num = 10
    voice_time = 2.0
    target_snr = 5.0
    sample_rate = 16000

    for noise_type in noise_types:

        dataset = SpeechEnhancementDataset(
            data_dir=data_dir,
            files=val_files,
            max_files=files_num,
            duration=voice_time,
            snr_db=target_snr,
            noise_type=noise_type,
        )

        # 为每个频段分别保存输入和输出误差
        band_results = {}

        for band in bands:
            band_results[band] = {
                "input": [],
                "output": [],
            }

        # 保存整体 MSE，用于检查四频段统计
        input_total_mse_values = []
        output_total_mse_values = []

        # 保存噪声能量
        true_noise_energy_values = []
        predicted_noise_energy_values = []

        # 两类相关性分别按样本保存，再分别求有效值的平均。
        noise_correlation_values = []

        # 保存预测噪声与干净语音的频带相关系数
        speech_noise_correlation_values = []

        for index in range(len(dataset)):

            # 获取带噪语音和干净语音
            mixture, clean = dataset[index]

            # 每条语音只进行一次模型推理
            with torch.no_grad():
                predicted = model(mixture.unsqueeze(0))

            # 转换为 NumPy 数组
            mixture_np = mixture.numpy()
            clean_np = clean.numpy()
            predicted_np = predicted.squeeze(0).numpy()

            # 真实加入的噪声
            true_noise = mixture_np - clean_np

            # 模型预测的噪声
            predicted_noise = mixture_np - predicted_np

            # 计算整体 MSE
            input_total_mse = calculate_mse(mixture_np, clean_np)
            output_total_mse = calculate_mse(predicted_np, clean_np)

            input_total_mse_values.append(input_total_mse)
            output_total_mse_values.append(output_total_mse)

            # 构造全零参考信号
            zeros = np.zeros_like(clean_np)

            # 真实噪声在 1000–3000 Hz 的能量
            true_noise_band_mse, _ = calculate_band_and_total_mse(
                predicted=true_noise,
                clean=zeros,
                sample_rate=sample_rate,
                low_freq=1000,
                high_freq=3000,
            )

            # 模型估计噪声在 1000–3000 Hz 的能量
            predicted_noise_band_mse, _ = calculate_band_and_total_mse(
                predicted=predicted_noise,
                clean=zeros,
                sample_rate=sample_rate,
                low_freq=1000,
                high_freq=3000,
            )
            true_noise_energy_values.append(true_noise_band_mse)
            predicted_noise_energy_values.append(predicted_noise_band_mse)

            # 计算降噪后的 1000–3000 Hz 重建误差 MSE
            output_error_band_mse, _ = calculate_band_and_total_mse(
                predicted=predicted_np,
                clean=clean_np,
                sample_rate=sample_rate,
                low_freq=1000,
                high_freq=3000,
            )

            # 计算干净语音在 1000–3000 Hz 的频带能量
            speech_band_energy, _ = calculate_band_and_total_mse(
                predicted=clean_np,
                clean=zeros,
                sample_rate=sample_rate,
                low_freq=1000,
                high_freq=3000,
            )

            # 计算预测噪声与干净语音之差的频带能量
            difference_band_energy, _ = calculate_band_and_total_mse(
                predicted=predicted_noise,
                clean=clean_np,
                sample_rate=sample_rate,
                low_freq=1000,
                high_freq=3000,
            )

            # 真实噪声 n=x-s，预测噪声 n_hat=x-s_hat；重建误差为 n-n_hat。
            noise_correlation = calculate_band_correlation_from_energies(
                true_noise_band_mse,
                predicted_noise_band_mse,
                output_error_band_mse,
            )
            noise_correlation_values.append(noise_correlation)

            # 另一项比较 n_hat 与干净语音 s，差值能量来自 n_hat-s。
            speech_noise_correlation = calculate_band_correlation_from_energies(
                predicted_noise_band_mse,
                speech_band_energy,
                difference_band_energy,
            )

            speech_noise_correlation_values.append(
                speech_noise_correlation
            )

            # 计算当前语音在四个频段的输入和输出 MSE
            for low_freq, high_freq in bands:

                input_band_mse, _ = calculate_band_and_total_mse(
                    predicted=mixture_np,
                    clean=clean_np,
                    sample_rate=sample_rate,
                    low_freq=low_freq,
                    high_freq=high_freq,
                )

                output_band_mse, _ = calculate_band_and_total_mse(
                    predicted=predicted_np,
                    clean=clean_np,
                    sample_rate=sample_rate,
                    low_freq=low_freq,
                    high_freq=high_freq,
                )

                band = (low_freq, high_freq)

                band_results[band]["input"].append(input_band_mse)
                band_results[band]["output"].append(output_band_mse)

        # 输出当前噪声类型的平均统计结果
        print(f"\n{'=' * 75}")
        print(f"噪声类型: {noise_type}")
        print(f"样本数量: {len(dataset)}")
        print(f"输入 SNR: {target_snr:.1f} dB")
        print(f"{'=' * 75}")

        print(
            f"{'频段':<18}"
            f"{'输入 MSE':>14}"
            f"{'输出 MSE':>14}"
            f"{'变化量':>14}"
        )

        print("-" * 75)

        input_band_sum = 0.0
        output_band_sum = 0.0

        for low_freq, high_freq in bands:

            band = (low_freq, high_freq)

            mean_input_mse = np.mean(
                band_results[band]["input"]
            )

            mean_output_mse = np.mean(
                band_results[band]["output"]
            )

            # 正数表示误差增加，负数表示误差减少
            mse_change = mean_output_mse - mean_input_mse

            input_band_sum += mean_input_mse
            output_band_sum += mean_output_mse

            band_name = f"{low_freq}-{high_freq} Hz"

            print(
                f"{band_name:<18}"
                f"{mean_input_mse:>14.6f}"
                f"{mean_output_mse:>14.6f}"
                f"{mse_change:>+14.6f}"
            )

        # 计算整体平均 MSE
        mean_input_total_mse = np.mean(input_total_mse_values)
        mean_output_total_mse = np.mean(output_total_mse_values)

        total_mse_change = (
            mean_output_total_mse - mean_input_total_mse
        )

        print("-" * 75)

        print(
            f"{'整体':<18}"
            f"{mean_input_total_mse:>14.6f}"
            f"{mean_output_total_mse:>14.6f}"
            f"{total_mse_change:>+14.6f}"
        )

        # 验证四个频段 MSE 之和等于整体 MSE
        input_check = np.allclose(
            input_band_sum,
            mean_input_total_mse,
        )

        output_check = np.allclose(
            output_band_sum,
            mean_output_total_mse,
        )

        print()
        print(f"输入频段能量检查: {input_check}")
        print(f"输出频段能量检查: {output_check}")

        mean_true_energy = np.mean(true_noise_energy_values)
        mean_predicted_energy = np.mean(predicted_noise_energy_values)

        print(f"\n噪声类型: {noise_type}")
        print("频段: 1000~3000 Hz")
        print(f"真实噪声平均频带能量: {mean_true_energy:.6f}")
        print(f"预测噪声平均频带能量: {mean_predicted_energy:.6f}")
        print(f"预测/真实能量比: {mean_predicted_energy / mean_true_energy:.2f}")
        # 分别对每条语音的有效系数求平均。
        valid_noise_correlations = np.asarray(noise_correlation_values)
        valid_noise_correlations = valid_noise_correlations[
            np.isfinite(valid_noise_correlations)
        ]

        if len(valid_noise_correlations) > 0:
            mean_noise_correlation = np.mean(valid_noise_correlations)
            print(
                "真实噪声与预测噪声平均相关系数: "
                f"{mean_noise_correlation:.4f}"
            )
            print(f"有效样本数量: {len(valid_noise_correlations)}")
        else:
            print("没有有效的频带相关系数")
        # 预测噪声与干净语音的相关性单独统计。
        valid_speech_correlations = np.asarray(
            speech_noise_correlation_values
        )

        valid_speech_correlations = valid_speech_correlations[
            np.isfinite(valid_speech_correlations)
        ]

        if len(valid_speech_correlations) > 0:
            mean_speech_correlation = np.mean(
                valid_speech_correlations
            )

            print("\n预测噪声与干净语音的相关性")
            print("频段: 1000–3000 Hz")
            print(
                "预测噪声与干净语音平均相关系数: "
                f"{mean_speech_correlation:.4f}"
            )
            print(f"有效样本数量: {len(valid_speech_correlations)}")
        else:
            print("没有有效的语音相关系数")
            
if __name__ == "__main__":
    main()
