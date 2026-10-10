
from pathlib import Path

import numpy as np
import soundfile as sf

"""
名称:混淆函数
目的:按照指定SNR,将噪声加入干净语音。
参数:
    clean: 干净语音,NumPy一维数组
    noise: 原始噪声,NumPy一维数组
    snr_db: 目标信噪比,单位dB
返回:
    mixture: 带噪语音
    scaled_noise: 缩放后的噪声
"""
def add_noise(clean, noise, snr_db):
    
    if clean.ndim != 1 or noise.ndim != 1:
        raise ValueError("输入必须是单通道一维数组")

    if len(clean) != len(noise):
        raise ValueError("语音和噪声长度必须一致")

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if clean_power <= 0 or noise_power <= 0:
        raise ValueError("语音和噪声功率必须大于0")

    scale = np.sqrt(
        clean_power /
        (noise_power * 10 ** (snr_db / 10))
    )

    scaled_noise = noise * scale
    mixture = clean + scaled_noise

    return mixture, scaled_noise
if __name__ == "__main__":
    # 1. 定位数据集
    BASE_DIR = Path(__file__).resolve().parent

    data_dir = BASE_DIR / "data" / "LibriSpeech" / "dev-clean"
    output_dir = BASE_DIR / "data" / "data_proc"
    output_dir.mkdir(exist_ok=True)

    # 2. 查找一条真实语音
    flac_files = sorted(data_dir.rglob("*.flac"))

    if not flac_files:
        raise FileNotFoundError("没有找到 FLAC 文件")

    audio_path = flac_files[0]

    # 3. 读取音频
    clean, sr = sf.read(audio_path)

    print("音频文件:", audio_path.name)
    print("采样率:", sr)
    print("音频形状:", clean.shape)
    print("时长:", len(clean) / sr, "秒")

    assert sr == 16000, "本次实验要求16 kHz"
    assert clean.ndim == 1, "本次实验要求单通道"

    # 4. 生成白噪声
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(clean))

    # 5. 利用混淆函数生成混合音频
    target_snr_db = 10.0
    mixture, scaled_noise = add_noise(
        clean,
        noise,
        snr_db=target_snr_db
    )

    # 6. 验证实际SNR
    actual_snr_db = 10 * np.log10(
        np.mean(clean ** 2) /
        np.mean(scaled_noise ** 2)
    )

    print(f"目标SNR: {target_snr_db:.2f} dB")
    print(f"实际SNR: {actual_snr_db:.2f} dB")

    # 7. 统一缩放，避免保存时削波
    peak = max(
        np.max(np.abs(clean)),
        np.max(np.abs(mixture))
    )
    gain = min(1.0, 0.99 / peak)

    # 8. 保存为WAV
    sf.write(
        output_dir / "clean.wav",
        clean * gain,
        sr,
        subtype="FLOAT"
    )

    sf.write(
        output_dir / "mixture.wav",
        mixture * gain,
        sr,
        subtype="FLOAT"
    )

    print("保存完成:", output_dir)

