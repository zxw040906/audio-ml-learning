from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from noise_type import generate_low_frequency_noise

def calculate_spectrum(audio, sr):
    """
    计算音频的单边幅度谱。
    """
    n = len(audio)

    # Hann窗减轻频谱泄漏
    window = np.hanning(n)
    windowed = audio * window

    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sr
    )

    magnitude = np.abs(spectrum)

    # 转换为dB，避免log(0)
    magnitude_db = 20 * np.log10(
        magnitude + 1e-10
    )

    return frequencies, magnitude_db


def calculate_low_frequency_ratio(signal, sample_rate=16000, cutoff=500):
    # 计算实数信号的单边 FFT
    spectrum = np.fft.rfft(signal)

    # 生成与 FFT 结果对应的频率坐标
    freqs = np.fft.rfftfreq(len(signal),1/sample_rate)

    # 计算各频率点的幅度平方
    power = np.abs(spectrum)**2

    # 修正单边频谱能量
    if len(signal) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2

    # 找出不高于截止频率的频率点
    low_mask = freqs<=cutoff

    # 计算低频能量占总能量的比例
    ratio = np.sum(power[low_mask])/np.sum(power)

    return ratio

"""
    根据 Parseval 定理，时域信号的平方和等于
    完整双边 FFT 幅度平方和除以原始信号长度 N:

        sum(|e[n]|^2) = (1/N) * sum(|E[k]|^2)

    这里已经将 rfft 的单边频谱修正为双边总能量，
    因此只需除以 len(error)，就能得到对应的时域误差能量。

    注意：这里除以的是原始信号长度，而不是 rfft 的频点数量。
    结果是误差能量，不是均方误差(MSE)。
"""
def calculate_band_and_total_mse(
    predicted,
    clean,
    sample_rate=16000,
    low_freq=1000,
    high_freq=3000,
):
    # 计算重建误差
    error = predicted - clean

    # 计算单边 FFT
    spectrum = np.fft.rfft(error)

    # 获取各 FFT 频点对应的频率
    freqs = np.fft.rfftfreq(len(error),1/sample_rate)

    # 计算功率谱
    power = np.abs(spectrum)**2

    # 将单边功率谱修正为完整频谱对应的能量
    if len(error) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2

    # 选取目标频段
    # 左闭右开，最后一个频段包含 Nyquist 频率
    if high_freq == sample_rate / 2:
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    else:
        band_mask = (freqs >= low_freq) & (freqs < high_freq)

    # 计算目标频段误差能量
    band_energy = np.sum(power[band_mask])/len(error)

    # 转换为每个采样点的均方误差
    band_mse = band_energy/len(error)

    #计算整体均方误差
    total_mse = np.mean(error ** 2)
    return band_mse ,total_mse

def main():
    #calculate_spectrum 测试
    # base_dir = Path(__file__).resolve().parent
    # output_dir = base_dir / "output"

    # names = [
    #     "eval_clean.wav",
    #     "eval_mixture.wav",
    #     "eval_predicted.wav",
    # ]

    # labels = [
    #     "Clean Speech",
    #     "Noisy Speech",
    #     "CNN Output",
    # ]

    # plt.figure(figsize=(12, 6))

    # for name, label in zip(names, labels):
    #     audio, sr = sf.read(output_dir / name)

    #     if audio.ndim != 1:
    #         raise ValueError("仅支持单通道音频")

    #     frequencies, magnitude_db = (
    #         calculate_spectrum(audio, sr)
    #     )

    #     plt.plot(
    #         frequencies,
    #         magnitude_db,
    #         label=label,
    #         alpha=0.75,
    #     )

    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude (dB)")
    # plt.title("Speech Spectrum Comparison")

    # # 先观察0~8000 Hz
    # plt.xlim(0, 8000)
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()

    # save_path = output_dir / "spectrum_comparison.png"

    # plt.savefig(save_path, dpi=150)
    # plt.show()

    # print("频谱图保存到:", save_path)

    #calculate_low_frequency_ratio 测试
    # sample_rate = 16000
    # length = 16000

    # rng = np.random.default_rng(42)
    # white_noise = rng.standard_normal(length)

    # low_frequency_noise = generate_low_frequency_noise(
    #     length=length,
    #     a=0.95,
    #     seed=42
    # )

    # white_ratio = calculate_low_frequency_ratio(white_noise)
    # low_ratio = calculate_low_frequency_ratio(low_frequency_noise)

    # print(f"白噪声低频能量占比: {white_ratio:.2%}")
    # print(f"低通噪声低频能量占比: {low_ratio:.2%}")

    #calculate_band_and_total_mse 测试
    rng = np.random.default_rng(42)

    clean = rng.standard_normal(16000)
    predicted = clean + 0.1 * rng.standard_normal(16000)

    bands = [
    (0, 500),
    (500, 1000),
    (1000, 3000),
    (3000, 8000),
    ]

    band_mse_sum = 0.0

    for low_freq, high_freq in bands:
        band_mse, total_mse = calculate_band_and_total_mse(
            predicted,
            clean,
            low_freq=low_freq,
            high_freq=high_freq,
        )

        band_mse_sum += band_mse

    print(f"四个频段 MSE 之和: {band_mse_sum:.8f}")
    print(f"整体 MSE: {total_mse:.8f}")
    print(f"是否一致: {np.allclose(band_mse_sum, total_mse)}")



   
if __name__ == "__main__":
    main()
