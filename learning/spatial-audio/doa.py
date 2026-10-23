"""GCC-PHAT 精细 TDOA 实验：多延迟回归 + 三种模式精度/耗时对照。"""

import numpy as np
from statistics import median
from time import perf_counter

from spatial_utils import (
    estimate_phat_delay,
    estimate_phat_delay_precise,
    tdoa_to_doa,
)


def add_noise_at_snr(signal, snr_db, rng):
    """
    给一维实数信号添加指定 SNR 的白噪声。

    参数：
        signal:
            原始信号。
        snr_db:
            目标信噪比，单位 dB。
        rng:
            NumPy 随机数生成器。

    返回：
        np.ndarray:
            加噪后的信号。
    """
    signal = np.asarray(signal, dtype=float)

    signal_rms = np.sqrt(np.mean(signal ** 2))

    if signal_rms == 0:
        raise ValueError("不能对全零信号指定有限 SNR")

    # 生成标准正态白噪声
    noise = rng.standard_normal(len(signal))

    # 计算当前噪声 RMS
    noise_rms = np.sqrt(np.mean(noise ** 2))

    # 计算目标噪声 RMS
    target_noise_rms = (
        signal_rms / (10 ** (snr_db / 20))
    )

    # 将噪声调整到目标幅度
    noise = noise / noise_rms * target_noise_rms

    return signal + noise


def simulate_delay_fft(signal, delay_samples):
    """
    使用零填充与 FFT 相位平移模拟带符号的分数采样延迟。

    仅用于当前无噪声仿真实验。有限长信号的频谱和边界仍有近似误差。
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("signal 必须是非空一维信号")
    if not np.isscalar(delay_samples) or not np.isfinite(delay_samples):
        raise ValueError("delay_samples 必须是有限数值")

    N = len(signal)
    pad = int(np.ceil(abs(delay_samples))) + 16
    padded = np.pad(signal, (pad, pad))

    # 奇数 FFT 长度，避免分数延迟下的 Nyquist 频点特殊处理
    if len(padded) % 2 == 0:
        padded = np.pad(padded, (0, 1))

    fft_length = len(padded)
    X = np.fft.fft(padded)
    frequencies = np.fft.fftfreq(fft_length)
    phase = np.exp(-2j * np.pi * frequencies * delay_samples)
    delayed_padded = np.fft.ifft(X * phase).real
    return delayed_padded[pad:pad + N]


def benchmark(callback, repeats=15):
    """返回包含算法完整 FFT + 搜索的单次运行时间中位数（毫秒）。"""
    for _ in range(3):
        callback()

    times = []
    for _ in range(repeats):
        start = perf_counter()
        callback()
        times.append((perf_counter() - start) * 1000)

    return median(times)


def main():
    sample_rate = 16000
    mic_distance = 0.1
    sound_speed = 343.0
    max_lag = int(np.ceil(mic_distance * sample_rate / sound_speed))

    rng = np.random.default_rng(42)
    clean = rng.standard_normal(4096)

    print("=== 多种理论延迟的回归测试 ===")
    print("理论延迟 | 整数+抛物线 | 精确估计 | 绝对误差 | 搜索状态")

    # 这些点分布在不同整数区间，避免 2.3324 恰好贴近 1/3 网格的巧合
    test_delays = [-3.42, -1.68, 0.0, 1.25, 2.27, 2.3324, 3.42, 4.30]

    for true_delay in test_delays:
        mic2 = simulate_delay_fft(clean, true_delay)

        parabolic = estimate_phat_delay(
            clean, mic2, max_lag, subsample=True
        )
        precise, status = estimate_phat_delay_precise(
            clean, mic2, max_lag,
            upsample=5, mode="auto", return_info=True,
        )
        print(
            f"{true_delay:+8.4f} | {parabolic:+11.6f} | "
            f"{precise:+9.6f} | {abs(precise - true_delay):.6f} | {status}"
        )

    print("\n=== 同一信号：不同粗定位方式的精度与总耗时 ===")
    true_angle = 30.0
    true_delay = (
        mic_distance * sample_rate * np.sin(np.deg2rad(true_angle))
        / sound_speed
    )
    mic2 = simulate_delay_fft(clean, true_delay)
    print(f"真实方向: {true_angle:.2f}°, 理论延迟: {true_delay:.6f} 点")
    print("模式             | 估计延迟 | 延迟误差 | DOA误差 | 耗时(ms) | 状态")

    modes = [
        ("仅整数+二分", "integer", 5),
        ("自动回退", "auto", 5),
        ("强制3倍+二分", "upsampled", 3),
        ("强制5倍+二分", "upsampled", 5),
    ]

    for label, mode, upsample in modes:
        def run():
            return estimate_phat_delay_precise(
                clean, mic2, max_lag,
                upsample=upsample, mode=mode, return_info=True,
            )

        estimated, status = run()
        elapsed = benchmark(run)
        doa = tdoa_to_doa(
            estimated, sample_rate, mic_distance, sound_speed
        )
        print(
            f"{label:<12} | {estimated:+.6f} | "
            f"{abs(estimated - true_delay):.6f} | "
            f"{abs(doa - true_angle):.4f}° | "
            f"{elapsed:8.3f} | {status}"
        )

    # 旧接口默认必须继续返回整数延迟
    old_integer = estimate_phat_delay(clean, mic2, max_lag)
    assert isinstance(old_integer, (int, np.integer))
    print(f"\n原 estimate_phat_delay() 兼容性检查通过：{old_integer} 点")
    print("注意：自动模式仅在整数细化失败时才会执行低倍率过采样。")

def noise_experiment():
    rng = np.random.default_rng(42)

    signal_length = 1024
    true_delay = 2.3324
    max_lag = 5

    clean = rng.standard_normal(signal_length)

    # 固定两路无噪声信号
    clean_mic1 = clean.copy()
    clean_mic2 = simulate_delay_fft(
        clean, true_delay
    )

    snr_levels = [30, 20, 10, 0, -5]
    trials = 30

    print(
        "SNR | 抛物线MAE | 精确估计MAE | "
        "整数细化 | 过采样细化 | 其他回退"
    )

    for snr_db in snr_levels:

        parabolic_errors = []
        precise_errors = []

        integer_success = 0
        upsample_success = 0
        fallback_count = 0

        for _ in range(trials):

            # 两路麦克风加入独立噪声
            mic1 = add_noise_at_snr(
                clean_mic1, snr_db, rng
            )
            mic2 = add_noise_at_snr(
                clean_mic2, snr_db, rng
            )

            # 传统抛物线插值
            parabolic_delay = estimate_phat_delay(
                mic1,
                mic2,
                max_lag,
                subsample=True,
            )

            # 新的精确估计器
            precise_delay, status = (
                estimate_phat_delay_precise(
                    mic1,
                    mic2,
                    max_lag,
                    upsample=5,
                    mode="auto",
                    return_info=True,
                )
            )

            parabolic_errors.append(
                abs(parabolic_delay - true_delay)
            )
            precise_errors.append(
                abs(precise_delay - true_delay)
            )

            if status == "integer_refined":
                integer_success += 1
            elif status == "upsampled_refined":
                upsample_success += 1
            else:
                fallback_count += 1

        # 多次试验的平均绝对误差
        parabolic_mae = np.mean(parabolic_errors)
        precise_mae = np.mean(precise_errors)

        print(
            f"{snr_db:+4d} | "
            f"{parabolic_mae:11.4f} | "
            f"{precise_mae:13.4f} | "
            f"{integer_success:8d} | "
            f"{upsample_success:10d} | "
            f"{fallback_count:8d}"
        )


def analyze_gcc_peak(mic1, mic2, max_lag):
    """提取整数延迟搜索区间内的 GCC-PHAT 峰值可信度特征。"""
    mic1 = np.asarray(mic1, dtype=float)
    mic2 = np.asarray(mic2, dtype=float)
    if mic1.ndim != 1 or mic2.ndim != 1 or len(mic1) == 0 or len(mic1) != len(mic2):
        raise ValueError("mic1 和 mic2 必须是长度相同的非空一维信号")
    if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag 必须是非负整数")

    n = len(mic1)
    limit = min(int(max_lag), n - 1)
    n_fft = 2 * n - 1
    x1 = np.fft.fft(mic1, n=n_fft)
    x2 = np.fft.fft(mic2, n=n_fft)
    cross_spectrum = x2 * np.conj(x1)
    phat = cross_spectrum / np.maximum(np.abs(cross_spectrum), 1e-12)
    gcc = np.fft.ifft(phat).real
    lags = np.arange(-limit, limit + 1)
    values = gcc[np.mod(lags, n_fft)]

    if not np.all(np.isfinite(values)):
        return {"peak_margin": 0.0, "normalized_peak": 0.0}

    main_index = int(np.argmax(values))
    main_peak = float(values[main_index])
    side_mask = np.ones(len(values), dtype=bool)
    side_mask[max(0, main_index - 1):min(len(values), main_index + 2)] = False
    sidelobes = values[side_mask]
    if sidelobes.size == 0:
        return {"peak_margin": 0.0, "normalized_peak": 0.0}

    competitor_peak = float(np.max(sidelobes))
    mean_sidelobe = float(np.mean(sidelobes))
    std_sidelobe = float(np.std(sidelobes))
    margin = main_peak - competitor_peak
    normalized = (main_peak - mean_sidelobe) / (std_sidelobe + 1e-12)
    if not np.isfinite(margin):
        margin = 0.0
    if not np.isfinite(normalized):
        normalized = 0.0
    return {"peak_margin": float(margin), "normalized_peak": float(normalized)}


def _mean_or_na(values):
    return f"{np.mean(values):.4f}" if values else "N/A"


def noise_failure_experiment():
    rng = np.random.default_rng(42)

    true_delay = 2.3324
    max_lag = 5

    signal_lengths = [256, 1024]
    snr_levels = [0, -5, -10, -15]

    trials = 100

    print(
        "长度 | SNR | 精确MAE | 整数错峰率 | 严重误差率 | 回退次数 | "
        "非严重/严重数 | margin(非严重/严重) | norm峰(非严重/严重)"
    )

    for signal_length in signal_lengths:

        clean = rng.standard_normal(signal_length)

        clean_mic1 = clean.copy()
        clean_mic2 = simulate_delay_fft(
            clean, true_delay
        )

        for snr_db in snr_levels:

            errors = []
            wrong_integer_count = 0
            gross_error_count = 0
            fallback_count = 0
            good_margins, bad_margins = [], []
            good_normalized, bad_normalized = [], []

            for _ in range(trials):

                mic1 = add_noise_at_snr(
                    clean_mic1, snr_db, rng
                )
                mic2 = add_noise_at_snr(
                    clean_mic2, snr_db, rng
                )

                # 1. 先检查整数粗定位
                integer_delay = estimate_phat_delay(
                    mic1,
                    mic2,
                    max_lag,
                    subsample=False,
                )

                # 2. 精细 TDOA 估计
                precise_delay, status = (
                    estimate_phat_delay_precise(
                        mic1,
                        mic2,
                        max_lag,
                        upsample=5,
                        mode="auto",
                        return_info=True,
                    )
                )

                error = abs(
                    precise_delay - true_delay
                )
                errors.append(error)
                peak = analyze_gcc_peak(mic1, mic2, max_lag)

                # 与最接近理论延迟的整数位置比较
                expected_integer = int(
                    np.rint(true_delay)
                )

                if integer_delay != expected_integer:
                    wrong_integer_count += 1

                # 严重误差阈值：1个采样点
                if error > 1.0:
                    gross_error_count += 1
                    bad_margins.append(peak["peak_margin"])
                    bad_normalized.append(peak["normalized_peak"])
                else:
                    good_margins.append(peak["peak_margin"])
                    good_normalized.append(peak["normalized_peak"])

                if status != "integer_refined":
                    fallback_count += 1

            mae = np.mean(errors)

            wrong_rate = (
                wrong_integer_count / trials * 100
            )
            gross_rate = (
                gross_error_count / trials * 100
            )

            print(
                f"{signal_length:4d} | "
                f"{snr_db:+4d} | "
                f"{mae:7.4f} | "
                f"{wrong_rate:9.1f}% | "
                f"{gross_rate:9.1f}% | "
                f"{fallback_count:4d} | "
                f"{len(good_margins):3d}/{len(bad_margins):3d} | "
                f"{_mean_or_na(good_margins)}/{_mean_or_na(bad_margins)} | "
                f"{_mean_or_na(good_normalized)}/{_mean_or_na(bad_normalized)}"
            )

if __name__ == "__main__":
    noise_failure_experiment()
    
# if __name__ == "__main__":
#     noise_experiment()

# if __name__ == "__main__":
#     main()
