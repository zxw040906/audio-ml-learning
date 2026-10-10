from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


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


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    names = [
        "eval_clean.wav",
        "eval_mixture.wav",
        "eval_predicted.wav",
    ]

    labels = [
        "Clean Speech",
        "Noisy Speech",
        "CNN Output",
    ]

    plt.figure(figsize=(12, 6))

    for name, label in zip(names, labels):
        audio, sr = sf.read(output_dir / name)

        if audio.ndim != 1:
            raise ValueError("仅支持单通道音频")

        frequencies, magnitude_db = (
            calculate_spectrum(audio, sr)
        )

        plt.plot(
            frequencies,
            magnitude_db,
            label=label,
            alpha=0.75,
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Speech Spectrum Comparison")

    # 先观察0~8000 Hz
    plt.xlim(0, 8000)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = output_dir / "spectrum_comparison.png"

    plt.savefig(save_path, dpi=150)
    plt.show()

    print("频谱图保存到:", save_path)


if __name__ == "__main__":
    main()
