import numpy as np


def simulate_delay(signal, delay_samples):
    """模拟一维信号的非负整数采样延迟，输出长度保持不变。"""

    if delay_samples < 0 or delay_samples > len(signal):
        raise ValueError("延迟必须在 0 到信号长度之间")

    delayed = np.zeros_like(signal)

    if delay_samples == 0:
        return signal.copy()

    # 使用变量完成延迟
    delayed[delay_samples:] = signal[:len(signal)-delay_samples]

    return delayed

def calc_estimate_tdoa(mic1,mic2,mic_distance,sample_rate,ms=False,sound_speed=343.0):
    """估计两路麦克风信号的带符号 TDOA。

    返回 t_B - t_A:
        正值:mic_b 比 mic_a 晚到达
        负值:mic_b 比 mic_a 先到达
        零值：估计的到达时间差为零
    采用物理约束下的普通互相关峰值估计。
"""

    mic1 = np.asarray(mic1)
    mic2 = np.asarray(mic2)

    if mic1.ndim != 1 or mic2.ndim != 1:
        raise ValueError("目前只支持一维麦克风信号")

    if len(mic1) == 0 or len(mic1) != len(mic2):
        raise ValueError("两路信号必须非空且长度相同")

    if mic_distance < 0 or sample_rate <= 0 or sound_speed <= 0:
        raise ValueError("麦克风间距不能为负，采样率和声速必须大于 0")

    # 计算完整互相关
    correlation = np.correlate(mic2, mic1, mode="full")

    # 构造延迟数组
    N = len(mic1)
    lags = np.arange(-(N - 1), N)

    # 根据物理间距计算最大延迟
    max_delay_samples = mic_distance * sample_rate / sound_speed
    max_lag = int(np.ceil(max_delay_samples))

    # 只保留物理范围内的互相关
    valid_mask = np.abs(lags) <= max_lag

    valid_lags = lags[valid_mask]
    valid_correlation = correlation[valid_mask]

    # 找到最大峰值对应的延迟
    peak_index = np.argmax(valid_correlation)
    estimated_delay = valid_lags[peak_index]

    # 转换成秒
    estimated_tdoa = estimated_delay / sample_rate

    if ms:
        return float(estimated_tdoa * 1000)

    return float(estimated_tdoa)
