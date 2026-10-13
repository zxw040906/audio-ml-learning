
from pathlib import Path

import numpy as np
import torch

from dataset import SpeechEnhancementDataset
from model import ResidualDenoisingCNN
from train import split_by_speaker
from analyze_spectrum import calculate_band_and_total_mse


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

        # 保存每条语音在 1000–3000 Hz 的噪声相关系数
        correlation_values = []
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
            input_total_mse = np.mean(
                (mixture_np - clean_np) ** 2
            )

            output_total_mse = np.mean(
                (predicted_np - clean_np) ** 2
            )

            input_total_mse_values.append(input_total_mse)
            output_total_mse_values.append(output_total_mse)

            # 构造全零参考信号
            zeros = np.zeros_like(clean_np)

            # 真实噪声在 1000–3000 Hz 的能量
            true_noise_band_mse,_= calculate_band_and_total_mse(
                predicted=true_noise,
                clean=zeros,
                low_freq=1000,
                high_freq=3000,
            )

            # 模型估计噪声在 1000–3000 Hz 的能量
            predicted_noise_band_mse,_= calculate_band_and_total_mse(
                predicted=predicted_noise,
                clean=zeros,
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

            # 根据 E_error = E_true + E_predicted - 2C 计算交叉能量
            cross_energy = (
                true_noise_band_mse
                + predicted_noise_band_mse
                - output_error_band_mse
            ) / 2

            # 计算相关系数的分母
            denominator = np.sqrt(
                true_noise_band_mse * predicted_noise_band_mse
            )

            # 计算当前语音的频带相关系数
            if denominator > 0:
                correlation = cross_energy / denominator
            else:
                correlation = np.nan

            # 保存当前语音的相关系数
            correlation_values.append(correlation)

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
        print("频段: 1000–3000 Hz")
        print(f"真实噪声平均频带能量: {mean_true_energy:.6f}")
        print(f"预测噪声平均频带能量: {mean_predicted_energy:.6f}")
        print(f"预测/真实能量比: {mean_predicted_energy / mean_true_energy:.2f}")
        # 计算10条语音的平均相关系数
        valid_correlations = np.asarray(correlation_values)
        valid_correlations = valid_correlations[
            np.isfinite(valid_correlations)
        ]

        if len(valid_correlations) > 0:
            mean_correlation = np.mean(valid_correlations)

            print(f"平均频带相关系数: {mean_correlation:.4f}")
            print(f"有效样本数量: {len(valid_correlations)}")
        else:
            print("没有有效的频带相关系数")

if __name__ == "__main__":
    main()
