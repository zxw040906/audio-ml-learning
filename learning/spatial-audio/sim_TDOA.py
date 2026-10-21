import numpy as np
from spatial_utils import simulate_delay, calc_estimate_tdoa

sample_rate = 16000
mic_distance = 0.1

# 麦克风 1 的目标信号
mic1 = np.zeros(30)
mic1[5] = 1.0

# 麦克风 2 晚收到目标声音 3 个采样点
true_delay = 3
mic2 = simulate_delay(mic1, true_delay)

# 在麦克风 2 中添加额外干扰
mic2[20] = 3.0


# 计算完整互相关
correlation = np.correlate(
    mic2,
    mic1,
    mode="full",
)

N = len(mic1)
lags = np.arange(-(N - 1), N)

# 练习 1：寻找全局最大互相关值所在的索引
peak_index = np.argmax(correlation)

# 练习 2：转换成实际采样延迟
unrestricted_delay = lags[peak_index]

# 通过物理约束互相关估计 TDOA
restricted_tdoa_ms = calc_estimate_tdoa(
    mic1,
    mic2,
    mic_distance=mic_distance,
    sample_rate=sample_rate,
    ms=True,
)

# 转换成采样点数，方便与无约束结果比较
restricted_delay = restricted_tdoa_ms / 1000 * sample_rate

print("无约束估计:", unrestricted_delay, "采样点")
print("物理约束估计:", restricted_delay, "采样点")
print("真实目标延迟:", true_delay, "采样点")
