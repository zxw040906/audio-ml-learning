from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from mix_audio import add_noise
from noise_type import generate_low_frequency_noise
from audio_utils import calculate_snr


class SpeechEnhancementDataset(Dataset):

    def __init__(
    self,
    data_dir,
    sample_rate=16000,
    duration=2.0,
    snr_db=5.0,
    max_files=20,
    files=None,
    noise_type="white"):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.snr_db = snr_db
        self.noise_type = noise_type
        if files is None:
            self.files = sorted(self.data_dir.rglob("*.flac"))
        else:
            self.files = sorted(Path(p) for p in files)

        if max_files is not None:
            self.files = self.files[:max_files]

        if not self.files:
            raise FileNotFoundError("没有找到音频文件")

    def __len__(self):
        # 数据集中有多少条语音
        return len(self.files)

    def __getitem__(self, index):
        # 只有访问这个索引时才读取音频
        audio_path = self.files[index]
        clean, sr = sf.read(audio_path)

        if sr != self.sample_rate:
            raise ValueError(
                f"采样率不正确: {audio_path}"
            )

        if clean.ndim != 1:
            raise ValueError("当前仅支持单通道音频")

        # 统一音频长度
        if len(clean) >= self.num_samples:
            clean = clean[:self.num_samples]
        else:
            clean = np.pad(
                clean,
                (0, self.num_samples - len(clean))
            )

        # 根据噪声类型生成噪声
        if self.noise_type == "white":

            # 保持原来的白噪声生成方式
            rng = np.random.default_rng(42 + index)

            noise = rng.standard_normal(
                self.num_samples
            )

        elif self.noise_type == "low_frequency":

            # 调用自己实现的低通噪声生成函数
            # 长度为 self.num_samples
            # 滤波系数 a 为 0.95
            # 随机种子为 42 + index
            noise = generate_low_frequency_noise(
                length=self.num_samples,
                a=0.95,
                seed=42+index
            )

        else:
            raise ValueError(
                f"不支持的噪声类型: {self.noise_type}"
            )

        mixture, _ = add_noise(
            clean,
            noise,
            snr_db=self.snr_db
        )

        # 转成PyTorch float32张量
        mixture = torch.from_numpy(
            mixture.astype(np.float32)
        )
        clean = torch.from_numpy(
            clean.astype(np.float32)
        )

        return mixture, clean


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent

    dataset = SpeechEnhancementDataset(
        data_dir=base_dir / "data" / "LibriSpeech" / "dev-clean",
        duration=2.0,
        snr_db=5.0,
        max_files=20,
        noise_type="white"
    )

    print("数据集大小:", len(dataset))

    mixture, clean = dataset[0]

    print("单条mixture形状:", mixture.shape)
    print("单条clean形状:", clean.shape)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch_mixture, batch_clean = next(
        iter(loader)
    )

    print("Batch mixture形状:", batch_mixture.shape)
    print("Batch clean形状:", batch_clean.shape)

    low_dataset = SpeechEnhancementDataset(
    data_dir=base_dir / "data" / "LibriSpeech" / "dev-clean",
    duration=2.0,
    snr_db=5.0,
    max_files=2,
    noise_type="low_frequency",
    )

    low_mixture, low_clean = low_dataset[0]

    print("低通噪声 mixture 形状:", low_mixture.shape)
    print("低通噪声 clean 形状:", low_clean.shape)


    clean_np = low_clean.numpy().astype(np.float64)
    mixture_np = low_mixture.numpy().astype(np.float64)
    input_snr = calculate_snr(clean_np, mixture_np)

    print(f"低通噪声输入 SNR: {input_snr:.2f} dB")
