import numpy as np

def parabolic_peak_offset(correlation, peak_index):
    """
    根据相关峰及其左右相邻点，估计亚采样点偏移量。

    参数：
        correlation:
            一维相关序列。

        peak_index:
            整数峰值在相关数组中的索引。

    返回：
        float:
            相对于整数峰值的小数偏移量。
            正值表示偏向右侧，负值表示偏向左侧。

    异常：
        ValueError:
            相关序列无效或索引越界。

        TypeError:
            峰值索引不是整数。

    注意：
        如果峰值位于序列边界，或插值分母接近零，
        返回 0.0，表示保持原来的整数峰值位置。
    """
    correlation = np.asarray(correlation)

    if correlation.ndim != 1 or len(correlation) == 0:
        raise ValueError("correlation 必须是非空一维数组")

    if not isinstance(peak_index, (int, np.integer)):
        raise TypeError("peak_index 必须是整数")

    if not 0 <= peak_index < len(correlation):
        raise ValueError("peak_index 超出数组范围")

    # 边界位置没有完整的左右邻点
    if peak_index == 0 or peak_index == len(correlation) - 1:
        return 0.0

    left = correlation[peak_index - 1]
    center = correlation[peak_index]
    right = correlation[peak_index + 1]

    denominator = 2 * (left - 2 * center + right)

    # 防止数值不稳定或除零
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0

    delta = (left - right) / denominator

    if not np.isfinite(delta):
        return 0.0

    return float(delta)

def _validate_mic_pair(mic1, mic2):
    """
    检查两路麦克风信号是否满足延迟估计要求。

    参数：
        mic1, mic2:
            两路一维麦克风信号。

    返回：
        tuple:
            转换为 NumPy 数组后的 (mic1, mic2)。

    异常：
        ValueError:
            信号不是一维、为空或长度不相同。
    """
    mic1 = np.asarray(mic1)
    mic2 = np.asarray(mic2)

    if mic1.ndim != 1 or mic2.ndim != 1:
        raise ValueError("目前只支持一维麦克风信号")

    if len(mic1) == 0 or len(mic1) != len(mic2):
        raise ValueError("两路信号必须非空且长度相同")

    return mic1, mic2


def simulate_delay(signal, delay_samples):
    """
    模拟一维信号的整数或分数采样延迟。

    参数：
        signal:
            原始一维实数信号，要求非空。

        delay_samples:
            带符号的采样延迟，单位为采样点。
            支持整数或小数。
            正值表示信号延后。
            负值表示信号提前。
            绝对值不能超过信号长度。

    返回：
        np.ndarray:
            延迟后的信号，长度与输入相同。
            超出原始信号范围的部分补零。

    异常：
        ValueError:
            信号不是非空一维实数数组，
            或延迟不是有限数值，
            或延迟绝对值超过信号长度。

    注意：
        1. 整数延迟使用数组切片，精确移动采样点。
        2. 小数延迟使用线性插值进行近似。
        3. 小数延迟可能改变信号的频谱特性，
           不等价于理想带限分数延迟。
        4. 整数延迟保留原数组数据类型；
           小数延迟通常返回浮点数组。
    """
    signal = np.asarray(signal)

    # 1. 检查输入信号
    if (
        signal.ndim != 1
        or len(signal) == 0
        or not np.issubdtype(signal.dtype, np.number)
        or np.iscomplexobj(signal)
    ):
        raise ValueError("signal 必须是非空一维实数数组")

    # 2. 检查延迟参数
    if (
        not np.isscalar(delay_samples)
        or not isinstance(
            delay_samples,
            (int, float, np.integer, np.floating)
        )
        or not np.isfinite(delay_samples)
    ):
        raise ValueError("delay_samples 必须是有限数值")

    N = len(signal)

    if abs(delay_samples) > N:
        raise ValueError("延迟绝对值不能超过信号长度")

    # 3. 整数延迟：直接使用数组切片
    if float(delay_samples).is_integer():
        k = int(delay_samples)

        delayed = np.zeros_like(signal)

        if k == 0:
            return signal.copy()

        if k > 0:
            # 正延迟：开头补零
            delayed[k:] = signal[:N-k]

        else:
            # 负延迟：信号提前，末尾补零
            q = -k
            delayed[:N-q] = signal[q:]

        return delayed

    # 4. 小数延迟：使用线性插值
    sample_positions = np.arange(N)

    # y[n] = x[n - delay_samples]
    shifted_positions = sample_positions - delay_samples

    delayed = np.interp(
        shifted_positions,
        sample_positions,
        signal,
        left=0.0,
        right=0.0,
    )

    return delayed



def estimate_normal_delay(mic1, mic2, max_lag):
    """
    使用物理约束范围内的普通互相关估计采样延迟。

    参数：
        mic1:
            第一通道麦克风信号。

        mic2:
            第二通道麦克风信号。

        max_lag:
            最大整数搜索延迟，单位为采样点。
            搜索范围为 [-max_lag, max_lag]。

    返回：
        int:
            带符号的整数采样延迟。
            正值表示 mic2 比 mic1 晚到达。
            负值表示 mic2 比 mic1 早到达。

    异常：
        ValueError:
            两路信号不满足要求，或 max_lag 为负。

        TypeError:
            max_lag 不是整数。

    注意：
        只计算允许搜索的延迟位置，
        避免完整互相关中不必要的计算。
    """
    mic1, mic2 = _validate_mic_pair(mic1, mic2)

    if not isinstance(max_lag, (int, np.integer)):
        raise TypeError("max_lag 必须是整数")

    if max_lag < 0:
        raise ValueError("max_lag 不能为负")

    N = len(mic1)

    # 搜索延迟不超过信号长度
    max_lag = min(max_lag, N - 1)

    lags = np.arange(-max_lag, max_lag + 1)
    correlation = []

    for k in lags:

        if k > 0:
            # mic2 比 mic1 晚到 k 个采样点
            value = np.dot(
                mic1[:N - k],
                mic2[k:]
            )

        elif k < 0:
            # mic2 比 mic1 早到 -k 个采样点
            q = -k

            value = np.dot(
                mic1[q:],
                mic2[:N - q]
            )

        else:
            # 两路信号不发生相对平移
            value = np.dot(mic1, mic2)

        correlation.append(value)

    correlation = np.asarray(correlation)

    peak_index = np.argmax(correlation)

    return int(lags[peak_index])


def estimate_phat_delay(
    mic1,
    mic2,
    max_lag,
    subsample=False,
):
    """
    使用 GCC-PHAT 估计两路信号的采样延迟。

    参数：
        mic1:
            第一通道麦克风信号。

        mic2:
            第二通道麦克风信号。

        max_lag:
            最大整数搜索延迟，单位为采样点。
        
        subsample:
            是否启用亚采样点峰值插值。
            False：返回整数延迟。
            True：返回浮点延迟。

    返回：
        int 或 float:
            subsample=False 时返回 int。
            subsample=True 时返回 float。

    异常：
        ValueError:
            两路信号不满足要求，或 max_lag 为负。

        TypeError:
            max_lag 不是整数。

    算法步骤：
        1. 对两路信号计算 FFT
        2. 计算互功率谱
        3. 进行 PHAT 幅度归一化
        4. IFFT 得到延迟域相关序列
        5. 在约束范围内搜索最大峰值

    注意：
        默认返回整数延迟；subsample=True 时使用抛物线插值。
    """
    mic1, mic2 = _validate_mic_pair(mic1, mic2)

    if not isinstance(max_lag, (int, np.integer)):
        raise TypeError("max_lag 必须是整数")

    if max_lag < 0:
        raise ValueError("max_lag 不能为负")

    N = len(mic1)
    max_lag = min(max_lag, N - 1)

    # 补零，使用足够的 FFT 长度
    n_fft = 2 * N - 1

    X1 = np.fft.fft(mic1, n=n_fft)
    X2 = np.fft.fft(mic2, n=n_fft)

    # 保持与普通互相关相同的延迟符号约定
    cross_spectrum = X2 * np.conj(X1)

    # PHAT 幅度归一化，避免除零
    phat = cross_spectrum / np.maximum(
        np.abs(cross_spectrum),
        1e-12,
    )

    # 逆 FFT，得到延迟域相关序列
    gcc = np.fft.ifft(phat).real

    # 把负延迟移动到数组前面
    gcc = np.concatenate((
        gcc[-(N - 1):],
        gcc[:N],
    ))

    lags = np.arange(-(N - 1), N)

    # 只在允许的整数延迟范围内搜索
    valid_mask = np.abs(lags) <= max_lag

    valid_lags = lags[valid_mask]
    valid_gcc = gcc[valid_mask]

    # 寻找最大相关峰值
    peak_index = np.argmax(valid_gcc)

    # 对应的整数采样延迟
    integer_delay = int(valid_lags[peak_index])

    # 默认保持整数估计，与旧版本兼容
    if not subsample:
        return integer_delay

    # 利用峰值左右邻点估计小数偏移
    delta = parabolic_peak_offset(
        valid_gcc,
        peak_index,
    )

    # 整数延迟 + 小数偏移
    estimated_delay = integer_delay + delta

    return float(estimated_delay)



def calc_estimate_tdoa(
    mic1,
    mic2,
    mic_distance,
    sample_rate,
    ms=False,
    sound_speed=343.0,
):
    """
    使用受限普通互相关估计双麦克风 TDOA。

    参数：
        mic1, mic2:
            两路一维麦克风信号。

        mic_distance:
            麦克风间距，单位 m。

        sample_rate:
            采样率，单位 Hz。

        ms:
            是否返回毫秒。
            False：返回秒。
            True：返回毫秒。

        sound_speed:
            声速，单位 m/s，默认 343.0。

    返回：
        float:
            带符号的 TDOA 时间差。
            正值表示 mic2 比 mic1 晚到达。
            负值表示 mic2 比 mic1 早到达。

    异常：
        ValueError:
            麦克风间距、采样率、声速或信号无效。

    注意：
        内部复用 estimate_normal_delay()，
        不再计算完整互相关。

        采用整数延迟搜索，因此 TDOA 的时间
        分辨率为 1 / sample_rate。
    """
    if mic_distance < 0:
        raise ValueError("麦克风间距不能为负")

    if sample_rate <= 0 or sound_speed <= 0:
        raise ValueError("采样率和声速必须大于 0")

    # 根据麦克风间距计算整数搜索边界
    max_lag = int(np.ceil(
        mic_distance * sample_rate / sound_speed
    ))

    # 复用优化后的普通互相关函数
    delay_samples = estimate_normal_delay(
        mic1,
        mic2,
        max_lag,
    )

    # 采样延迟转换成时间差，单位秒
    tdoa = delay_samples / sample_rate

    if ms:
        return float(tdoa * 1000)

    return float(tdoa)


def tdoa_to_doa(
    delay_samples,
    sample_rate,
    mic_distance,
    sound_speed=343.0,
):
    """
    将采样延迟转换为双麦克风 DOA 方向角。

    适用条件：
        双麦克风线性阵列。
        远场平面波模型。
        角度相对于阵列正侧面定义。

    参数：
        delay_samples:
            带符号的采样延迟，支持整数或浮点数。
            正值表示 mic2 比 mic1 晚接收到声音。

        sample_rate:
            采样率，单位 Hz。

        mic_distance:
            麦克风间距，单位 m。

        sound_speed:
            声速，单位 m/s，默认 343.0。

    返回：
        float:
            DOA 方向角，单位度。
            范围为 [-90, 90]。

    异常：
        ValueError:
            物理参数无效，或估计延迟超出物理范围。

    注意：
        输入单位是采样点，而不是秒。
        双麦克风 TDOA 存在前后镜像方向歧义。
    """
    if sample_rate <= 0:
        raise ValueError("采样率必须大于 0")

    if mic_distance <= 0:
        raise ValueError("麦克风间距必须大于 0")

    if sound_speed <= 0:
        raise ValueError("声速必须大于 0")

    # 采样点转换为时间差（秒）
    tdoa = delay_samples / sample_rate

    # 根据远场几何关系计算 sin(theta)
    sin_theta = sound_speed * tdoa / mic_distance

    # 检查反正弦函数的定义域
    if not -1 <= sin_theta <= 1:
        raise ValueError(
            "估计延迟超出物理允许范围，无法计算 DOA"
        )

    # 反正弦返回弧度，再转换为角度
    doa_deg = np.degrees(np.arcsin(sin_theta))

    return float(doa_deg)


# =============================================================
# 精细 GCC-PHAT：整数粗定位 + 导数二分，必要时低倍率过采样
# =============================================================

def _compute_phat_spectrum(mic1, mic2):
    """计算一次 PHAT 频谱；延迟符号：mic2 晚于 mic1 为正。"""
    mic1, mic2 = _validate_mic_pair(mic1, mic2)

    if not (np.issubdtype(mic1.dtype, np.number)
            and np.issubdtype(mic2.dtype, np.number)
            and not np.iscomplexobj(mic1)
            and not np.iscomplexobj(mic2)):
        raise ValueError("麦克风信号必须是实数数组")

    mic1 = mic1.astype(float, copy=False)
    mic2 = mic2.astype(float, copy=False)
    if not (np.all(np.isfinite(mic1)) and np.all(np.isfinite(mic2))):
        raise ValueError("麦克风信号不能含有 NaN 或 Inf")

    n_fft = 2 * len(mic1) - 1
    x1 = np.fft.fft(mic1, n_fft)
    x2 = np.fft.fft(mic2, n_fft)
    cross = x2 * np.conj(x1)

    magnitude = np.abs(cross)
    if not np.any(magnitude > 0):
        raise ValueError("两路信号的互功率谱为零，无法估计延迟")

    return cross / np.maximum(magnitude, 1e-12)


def _phat_integer_peak(phat, max_lag):
    """从原始 IFFT 的物理受限网格寻找整数峰值。"""
    gcc = np.fft.ifft(phat).real
    lags = np.arange(-max_lag, max_lag + 1)
    return float(lags[np.argmax(gcc[lags % len(gcc)])])


def _phat_upsampled_peak(phat, max_lag, upsample):
    """频谱居中补零，获得 1 / upsample 采样点间隔的相关网格。"""
    n_fft = len(phat)  # 2*N-1，始终为奇数
    centered = np.fft.fftshift(phat)
    pad = (upsample - 1) * n_fft // 2
    extended = np.pad(centered, (pad, pad))
    gcc = np.fft.ifft(np.fft.ifftshift(extended)).real * upsample

    ticks = np.arange(-max_lag * upsample, max_lag * upsample + 1)
    peak_tick = ticks[np.argmax(gcc[ticks % len(gcc)])]
    return float(peak_tick / upsample)


def _phat_refine_bisection(
    phat, center, half_width, max_lag, tolerance, max_iterations
):
    """搜索相关函数导数由正变负的局部驻点；不能建立区间时返回 None。"""
    frequency = np.fft.fftfreq(len(phat))
    omega = 2j * np.pi * frequency
    weights = omega * phat
    scale = len(phat)

    def derivative(delay):
        return float(np.real(np.dot(weights, np.exp(omega * delay))) / scale)

    left = max(-max_lag, center - half_width)
    right = min(max_lag, center + half_width)
    if left >= right:
        return None

    # 此条件适用于局部单峰区间；复杂多峰情况下不保证全局最优。
    if not (derivative(left) > 0 and derivative(right) < 0):
        return None

    for _ in range(max_iterations):
        if right - left <= tolerance:
            break
        mid = (left + right) / 2
        if derivative(mid) > 0:
            left = mid
        else:
            right = mid

    return float((left + right) / 2)


def estimate_phat_delay_precise(
    mic1,
    mic2,
    max_lag,
    upsample=5,
    mode="auto",
    tolerance=1e-5,
    max_iterations=25,
    return_info=False,
):
    """
    精细 GCC-PHAT 延迟估计，返回带符号的浮点采样延迟。

    mode:
        "auto"      先整数粗定位 + 导数二分；失败才用低倍率过采样回退。
        "integer"   只采用整数粗定位 + 导数二分，不启用过采样。
        "upsampled" 强制低倍率过采样粗定位 + 导数二分（用于对照实验）。

    upsample:
        回退或强制过采样的倍率，必须为 >=3 的奇数（如 3、5、7）。
        对于 mode="auto" 可设为 None，表示禁用过采样回退。

    return_info:
        True 时返回 (延迟, 状态字符串)：
        "integer_refined"、"upsampled_refined" 表示完成局部细化；
        "integer_grid"、"upsampled_grid" 表示回退到粗搜索网格。

    注意：
        只有局部搜索失败时，auto 模式才执行较长的过采样 IFFT。
        搜索数值精度不等于真实声源 TDOA 精度。
        噪声、混响和多峰情况下不保证找到全局主峰。
    """
    if not isinstance(max_lag, (int, np.integer)):
        raise TypeError("max_lag 必须是整数")
    if max_lag < 0:
        raise ValueError("max_lag 不能为负")
    if mode not in ("auto", "integer", "upsampled"):
        raise ValueError("mode 必须是 auto、integer 或 upsampled")

    if upsample is not None and (
        not isinstance(upsample, (int, np.integer))
        or upsample < 3
        or upsample % 2 == 0
    ):
        raise ValueError("upsample 必须是 >=3 的奇数或 None")
    if mode == "upsampled" and upsample is None:
        raise ValueError("mode='upsampled' 需要有效的 upsample 倍率")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance 必须是有限正数")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations 必须是正整数")

    phat = _compute_phat_spectrum(mic1, mic2)
    max_lag = min(int(max_lag), (len(phat) - 1) // 2)

    def pack(estimate, status):
        estimate = float(estimate)
        return (estimate, status) if return_info else estimate

    if mode in ("auto", "integer"):
        coarse = _phat_integer_peak(phat, max_lag)
        refined = _phat_refine_bisection(
            phat, coarse, 0.5, max_lag, tolerance, max_iterations
        )
        if refined is not None:
            return pack(refined, "integer_refined")
        if mode == "integer" or upsample is None:
            return pack(coarse, "integer_grid")

    # auto 模式的回退分支，或强制使用低倍率过采样的对照分支。
    coarse = _phat_upsampled_peak(phat, max_lag, upsample)
    refined = _phat_refine_bisection(
        phat, coarse, 1 / upsample, max_lag, tolerance, max_iterations
    )
    if refined is not None:
        return pack(refined, "upsampled_refined")
    return pack(coarse, "upsampled_grid")
