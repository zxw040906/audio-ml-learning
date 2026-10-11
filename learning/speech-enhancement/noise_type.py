import numpy as np


def generate_low_frequency_noise(length, a=0.95, seed=42):
    """
    生成低通滤波后的随机噪声。

    参数:
        length: 需要生成的采样点数
        a: 一阶滤波器系数
        seed: 随机种子

    返回:
        noise: 长度为 length 的 NumPy 数组
    """
    # 1: 使用 seed 创建 NumPy 随机数生成器
    rng = np.random.default_rng(seed)
    # 2: 生成 length 个标准正态分布随机数
    white = rng.standard_normal(length)
    # 3: 创建长度为 length 的全零输出数组
    noise = np.zeros(length)
    # 4: 使用 for 循环实现递推公式
    # noise[n] = a * noise[n-1] + (1-a) * white[n]
    # 注意 n=0 时的初始状态
    for n in range(length):
        if n==0:
            noise[0] = (1-a) * white[n]
        else:
            noise[n] = a * noise[n-1] + (1-a) * white[n]
    # 5: 返回 noise
    return noise

def scale_noise_to_snr(clean, noise, snr_db):
    """
    根据干净语音的功率，把噪声缩放到目标 SNR。

    参数:
        clean: 干净语音数组
        noise: 原始噪声数组
        snr_db: 目标 SNR,单位 dB

    返回:
        scaled_noise: 缩放后的噪声数组
    """

    # 计算干净语音的平均功率
    clean_power = np.mean(clean**2)

    # 计算原始噪声的平均功率
    noise_power = np.mean(noise**2)

    # 根据目标 SNR 计算需要的噪声功率
    target_noise_power = clean_power/(pow(10,snr_db/10))

    # 计算幅度缩放系数
    scale = np.sqrt(target_noise_power/noise_power)

    # 返回缩放后的噪声
    return noise * scale

if __name__ == "__main__":

    # sr = 16_000
    # #低频白噪声
    # noise = generate_low_frequency_noise(length=sr)

    # # 1：计算单边 FFT
    # spectrum = np.fft.rfft(noise)
    # # 2：生成频率坐标
    # freqs = np.fft.rfftfreq(len(noise),d=1/sr)

    # # 3：计算幅度谱
    # magnitude = np.abs(spectrum)

    # print("噪声长度:", len(noise))
    # print("频谱长度:", len(spectrum))
    # print("频率坐标长度:", len(freqs))
    # print("最低频率:", freqs[0])
    # print("最高频率:", freqs[-1])

    # # 创建布尔数组，选出 0～500 Hz 对应的频率点
    # low_mask = freqs <= 500

    # # 计算频谱中各频率点的幅度平方
    # """
    # 因为rfft只返回了非负频率,但是对于实数信号，完整 FFT 的正频率和负频率具有共轭对称性：
    # X[-k]=X[k]^* ,因此实际上需要处理:
    # 当原始信号长度为偶数时,power[1:-1] *= 2。
    # 当原始信号长度为奇数时,power[1:] *= 2。
    # 这样求出来的才是真正的能量
    # """
    # power = magnitude ** 2
    # one_sided_power = power.copy()
    # one_sided_power[1:-1] *=2
    # # 计算选中频率点的平方和
    # low_power = np.sum(one_sided_power[low_mask])
    # # 计算完整单边频谱能量
    # total_power = np.sum(one_sided_power)

    # print("低频能量:", low_power)
    # print("完整单边频谱能量:", total_power)

    # low_ratio = low_power / total_power
    # print(f"低频白噪声低频能量占比:{low_ratio * 100:.2f}%")

    # # 生成白噪声
    # rng = np.random.default_rng(42)
    # white_noise = rng.standard_normal(sr)

    # # 计算单边 FFT
    # white_spectrum = np.fft.rfft(white_noise)

    # # 计算幅度谱
    # white_magnitude = np.abs(white_spectrum)

    # # 计算每个频率点的幅度平方，注意这里不要 np.sum()
    # white_power = white_magnitude ** 2

    # # 复制数组，避免修改原始数据
    # one_sided_white_power = white_power.copy()

    # # 修正单边频谱能量
    # one_sided_white_power[1:-1] *= 2

    # # 计算 0～500 Hz 的能量
    # low_white_power = np.sum(one_sided_white_power[low_mask])

    # # 计算全部频率的能量
    # total_white_power = np.sum(one_sided_white_power)

    # # 计算低频能量占比
    # low_white_ratio = low_white_power / total_white_power

    # print(f"白噪声低频能量占比: {low_white_ratio * 100:.2f}%")

    clean = np.array([1.0, -1.0, 1.0, -1.0])
    noise = np.array([1.0, 1.0, -1.0, -1.0])

    scaled_noise = scale_noise_to_snr(clean, noise, 5)

    input_snr = 10 * np.log10(
        np.mean(clean ** 2) / np.mean(scaled_noise ** 2)
    )

    print(f"实际 SNR: {input_snr:.2f} dB")
