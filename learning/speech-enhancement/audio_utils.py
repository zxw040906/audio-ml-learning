"""单通道音频实验共用的数值分析函数。"""

import numpy as np
import random

from collections import defaultdict

def sample_files_by_speaker(test_files, test_dir, num_speakers=10, seed=42):
    """
    从不同说话人中各抽取一条语音。

    参数：
        test_files: 测试音频路径列表
        test_dir: test-clean 根目录
        num_speakers: 希望抽取的说话人数量
        seed: 随机种子，用于复现实验

    返回：
        selected_files: 选中的音频路径列表
    """

    # 第一步：按照说话人 ID 分组
    speaker_to_files = defaultdict(list)

    for file_path in test_files:
        # 例如：
        # test-clean/1089/134686/1089-134686-0000.flac
        speaker_id = file_path.relative_to(test_dir).parts[0]

        # 练习 1：将当前文件加入对应说话人的列表
        speaker_to_files[speaker_id].append(file_path)

    # 第二步：获取全部说话人 ID
    speaker_ids = sorted(speaker_to_files.keys())

    print("测试集说话人总数:", len(speaker_ids))

    # 使用独立的随机数生成器，不影响数据集的噪声生成
    rng = random.Random(seed)

    # 如果可用说话人不足，则使用全部说话人
    actual_num = min(num_speakers, len(speaker_ids))

    # 第三步：随机选择不同说话人
    selected_speakers = rng.sample(
        speaker_ids,
        actual_num,
    )

    # 第四步：每位说话人随机选一条语音
    selected_files = []

    for speaker_id in selected_speakers:
        files = sorted(speaker_to_files[speaker_id])

        # 练习 2：从该说话人的音频中随机抽取一条
        selected_file = rng.choice(files)
        
        selected_files.append(selected_file)

    return selected_files


def calculate_mse(predicted, clean):
    """计算形状相同信号的整体均方误差，返回幅值平方单位的标量。"""
    predicted = np.asarray(predicted)
    clean = np.asarray(clean)
    if predicted.shape != clean.shape:
        raise ValueError("预测信号和参考信号的形状必须一致")
    if predicted.size == 0:
        raise ValueError("信号不能为空")
    return float(np.mean((predicted - clean) ** 2))


def calculate_band_and_total_mse(
    predicted,
    clean,
    sample_rate=16000,
    low_freq=1000,
    high_freq=3000,
):
    """返回重建误差在指定频带及全频带的均方值。

    参数是等形状的一维信号、采样率（Hz）及频带边界（Hz）。频带左闭右开；
    上界为 Nyquist 时包含该端点。使用 rFFT 单边功率补偿和 Parseval 归一化，
    频带值与整体值单位均为信号幅值的平方；频带完整覆盖时各频带值之和等于
    整体 MSE。这里只适用于同采样率、时间对齐的实值信号。
    """
    predicted = np.asarray(predicted)
    clean = np.asarray(clean)
    if predicted.shape != clean.shape:
        raise ValueError("预测信号和参考信号的形状必须一致")
    if predicted.ndim != 1:
        raise ValueError("仅支持一维单通道信号")
    if predicted.size == 0:
        raise ValueError("信号不能为空")
    if sample_rate <= 0:
        raise ValueError("采样率必须大于 0")
    nyquist = sample_rate / 2
    if low_freq < 0 or high_freq <= low_freq or high_freq > nyquist:
        raise ValueError("频带边界必须满足 0 <= low_freq < high_freq <= Nyquist")

    error = predicted - clean
    spectrum = np.fft.rfft(error)
    frequencies = np.fft.rfftfreq(error.size, d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    if error.size % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2

    if high_freq == nyquist:
        band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    else:
        band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
    band_mse = np.sum(power[band_mask]) / (error.size ** 2)
    total_mse = np.mean(error ** 2)
    return float(band_mse), float(total_mse)


def calculate_band_energy(signal, sample_rate, low_freq, high_freq):
    """计算一维实值信号在指定频带内的均方能量。

    采样率及频带边界单位为 Hz，区间左闭右开（Nyquist 上界包含）。
    返回值与对应频带的按样本归一化能量相同，单位为幅值平方。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("仅支持一维单通道信号")
    energy, _ = calculate_band_and_total_mse(
        predicted=signal,
        clean=np.zeros_like(signal),
        sample_rate=sample_rate,
        low_freq=low_freq,
        high_freq=high_freq,
    )
    return energy


def calculate_band_correlation_from_energies(
    energy_a, energy_b, difference_energy
):
    """由相同频带、采样率和归一化下的三个均方能量求归一化内积。

    `difference_energy` 对应 A-B。由 C=(E_a+E_b-E_(a-b))/2 得到交叉项，
    再返回 C/sqrt(E_a E_b)。不去均值，因此不是 Pearson 相关系数；任一
    输入能量为零或负数时返回 NaN。
    """
    energy_a = float(energy_a)
    energy_b = float(energy_b)
    if energy_a <= 0 or energy_b <= 0:
        return np.nan
    cross_energy = (
        energy_a + energy_b - float(difference_energy)
    ) / 2
    return cross_energy / np.sqrt(energy_a * energy_b)


def calculate_band_correlation(
    signal_a, signal_b, sample_rate, low_freq, high_freq
):
    """计算等形状一维信号在指定频带内的归一化内积（非 Pearson 系数）。"""
    signal_a = np.asarray(signal_a)
    signal_b = np.asarray(signal_b)
    if signal_a.shape != signal_b.shape:
        raise ValueError("两个信号的形状必须一致")
    if signal_a.ndim != 1:
        raise ValueError("仅支持一维单通道信号")
    energy_a = calculate_band_energy(
        signal_a, sample_rate, low_freq, high_freq
    )
    energy_b = calculate_band_energy(
        signal_b, sample_rate, low_freq, high_freq
    )
    difference_energy = calculate_band_energy(
        signal_a - signal_b, sample_rate, low_freq, high_freq
    )
    return calculate_band_correlation_from_energies(
        energy_a, energy_b, difference_energy
    )


def calculate_snr(clean, estimate):
    """计算参考信号与估计信号之间的参考波形 SNR，单位为 dB。

    定义为 10 log10(mean(clean^2) / mean((estimate-clean)^2))。
    `clean`、`estimate` 可以是形状相同的一维信号或 `[B, T]` 批次；
    批次返回每条样本各自的 SNR。计算使用 float64。零误差返回正无穷；
    静音参考且零误差返回 NaN，静音参考但误差非零返回负无穷。
    """
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if clean.shape != estimate.shape:
        raise ValueError("参考信号和估计信号的形状必须一致")
    if clean.ndim not in (1, 2):
        raise ValueError("仅支持一维信号或形状为 [B, T] 的批次")
    axis = -1
    signal_power = np.mean(clean ** 2, axis=axis)
    error_power = np.mean((estimate - clean) ** 2, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * np.log10(signal_power / error_power)
    if clean.ndim == 1:
        return float(snr)
    return snr


def calculate_snr_improvement(clean, noisy, enhanced):
    """计算 SNR 提升，即增强后 SNR 减去输入带噪 SNR，单位 dB。"""
    return calculate_snr(clean, enhanced) - calculate_snr(clean, noisy)


__all__ = [
    "calculate_mse",
    "calculate_band_and_total_mse",
    "calculate_band_energy",
    "calculate_band_correlation",
    "calculate_band_correlation_from_energies",
    "calculate_snr",
    "calculate_snr_improvement",
    "sample_files_by_speaker",
]
