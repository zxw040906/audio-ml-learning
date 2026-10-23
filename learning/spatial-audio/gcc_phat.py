import numpy as np
from spatial_utils import (
    simulate_delay,
    estimate_normal_delay,
    estimate_phat_delay,
)

# 实验次数
num_trials = 100

# 真实延迟
true_delay = 3

# 麦克风距离
mic_distance = 0.1

# 声速
sound_speed = 343.0

# 噪声强度
noise_levels = [0.5, 1.0, 2.0, 3.0, 5.0]

# 信号长度(s)   `
duration_seconds = [0.004, 0.016, 0.064, 0.256]

# 采样率
sample_rate = 16000

# 两个麦克风之间允许搜索的最大整数采样延迟
max_lag = int(np.ceil(
    mic_distance * sample_rate / sound_speed
))

for duration_second in duration_seconds:
    # 根据音频时长计算采样点数
    signal_length = int(duration_second * sample_rate)

    print(f"\n===== 信号时长: {duration_second * 1000:.0f} ms =====")
    for noise_level in noise_levels:
    # 每个噪声强度使用相同的随机种子
        rng = np.random.default_rng(42)

        # 每个噪声强度都要重新清空结果
        normal_estimates = []
        phat_estimates = []

        # 每个噪声强度重复 num_trials 次
        for trial in range(num_trials):

            clean = rng.standard_normal(signal_length)

            noise1 = rng.standard_normal(signal_length)
            noise2 = rng.standard_normal(signal_length)

            mic1 = clean + noise_level * noise1
            mic2 = simulate_delay(clean, true_delay) + (
                noise_level * noise2
            )

            normal_delay = estimate_normal_delay(
                mic1, mic2, max_lag
            )

            phat_delay = estimate_phat_delay(
                mic1, mic2, max_lag
            )

            normal_estimates.append(normal_delay)
            phat_estimates.append(phat_delay)

        # 下面的统计代码应放在内层循环结束后
        normal_estimates = np.asarray(normal_estimates)
        phat_estimates = np.asarray(phat_estimates)

        # 计算 MAE
        normal_mae = np.mean(
            np.abs(normal_estimates - true_delay)
        )
        phat_mae = np.mean(
            np.abs(phat_estimates - true_delay)
        )

        # 计算完全正确率
        normal_accuracy = np.mean(
            normal_estimates == true_delay
        )
        phat_accuracy = np.mean(
            phat_estimates == true_delay
        )

        print(f"\n噪声系数: {noise_level}")
        print(f"普通互相关 MAE: {normal_mae:.3f} 点")
        print(f"GCC-PHAT MAE: {phat_mae:.3f} 点")
        print(f"普通互相关正确率: {normal_accuracy:.1%}")
        print(f"GCC-PHAT 正确率: {phat_accuracy:.1%}")
