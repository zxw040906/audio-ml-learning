from pathlib import Path

import numpy as np
import torch

from dataset import SpeechEnhancementDataset
from model import ResidualDenoisingCNN
from audio_utils import calculate_mse, calculate_snr, sample_files_by_speaker

def main():
    # 获取当前脚本所在目录
    base_dir = Path(__file__).resolve().parent

    # 使用独立测试集，不参与训练和模型选择
    test_dir = base_dir / "data" / "LibriSpeech" / "test-clean"
    model_path = base_dir / "output" / "best_residual_model.pth"

    # 查找独立测试集中的所有音频
    test_files = sorted(test_dir.rglob("*.flac"))
    #此处使用分层抽样确保测试评估客观可信
    selected_files = sample_files_by_speaker(
    test_files=test_files,
    test_dir=test_dir,
    num_speakers=10,
    seed=42,
    )
    print("抽样后音频数量:", len(selected_files))
    print("抽样文件:")
    for path in selected_files:
        print(path.relative_to(test_dir))

    # 加载已经训练好的残差 CNN
    model = ResidualDenoisingCNN()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()
    #噪声列表
    noise_types = ["white","low_frequency"]
    # 检查分层抽样覆盖了多少位说话人
    speaker_ids = set()

    for file_path in selected_files:
        speaker_id = file_path.relative_to(test_dir).parts[0]
        speaker_ids.add(speaker_id)

    print("测试语音数量:", len(selected_files))
    print("不同说话人数量:", len(speaker_ids))
    print("说话人 ID:", sorted(speaker_ids))

    assert len(selected_files) == len(speaker_ids)

    for noise_type in noise_types:
        # 先使用少量样本测试流程
        dataset = SpeechEnhancementDataset(
            data_dir=test_dir,
            files=selected_files,
            max_files=len(selected_files),
            duration=2.0,
            snr_db=5.0,
            noise_type=noise_type,
        )
        print(f"当前评估{noise_type}类型噪声:")
        input_snr_values = []
        output_snr_values = []
        snr_improvements = []

        input_mse_values = []
        output_mse_values = []

        for index in range(len(dataset)):
            # 先检查第一条语音
            mixture, clean = dataset[index]

            # 补全模型输入
            with torch.no_grad():
                predicted = model(
                    mixture.unsqueeze(0)
                ).squeeze(0)

            # 将输入,参考,预测语音转换为 NumPy
            predicted_np = predicted.numpy()
            mixture_np = mixture.numpy()
            clean_np = clean.numpy()

            # 检查模型输出与干净语音的均方误差
            input_mse = calculate_mse(mixture_np,clean_np)
            output_mse = calculate_mse(predicted_np, clean_np)

            input_snr = calculate_snr(clean_np,mixture_np)
            output_snr = calculate_snr(clean_np,predicted_np)

            # 计算 SNR 改善量
            snr_improvement = output_snr - input_snr
            input_snr_values.append(input_snr)
            output_snr_values.append(output_snr)
            snr_improvements.append(snr_improvement)

            input_mse_values.append(input_mse)
            output_mse_values.append(output_mse)

            print(f"第{index+1}条语音输入 SNR: {input_snr:.2f} dB")
            print(f"第{index+1}条语音输出 SNR: {output_snr:.2f} dB")
            print(f"第{index+1}条语音SNR 改善量: {snr_improvement:+.2f} dB")
            print(f"第{index+1}条语音输出 MSE:{output_mse:.6f}\n")
        
        # 计算所有测试语音的平均指标
        mean_input_snr = np.mean(input_snr_values)
        mean_output_snr = np.mean(output_snr_values)

        # 练习 1：计算平均 SNR 改善量
        mean_improvement = np.mean(snr_improvements)

        mean_input_mse = np.mean(input_mse_values)
        mean_output_mse = np.mean(output_mse_values)

        # 练习 2：统计 SNR 改善量大于 0 的样本数量
        improved_count = np.sum(
            np.asarray(snr_improvements) > 0
        )

        # 练习 3：计算改善比例
        improved_ratio = improved_count / len(dataset)

        print(f"\n噪声类型: {noise_type}")
        print(f"样本数量: {len(dataset)}")
        print(f"平均输入 SNR: {mean_input_snr:.2f} dB")
        print(f"平均输出 SNR: {mean_output_snr:.2f} dB")
        print(f"平均 SNR 改善: {mean_improvement:+.2f} dB")
        print(f"平均输入 MSE: {mean_input_mse:.6f}")
        print(f"平均输出 MSE: {mean_output_mse:.6f}")
        print(f"改善样本数量: {improved_count}/{len(dataset)}")
        print(f"改善样本比例: {improved_ratio:.1%}")


if __name__ == "__main__":
    main()
