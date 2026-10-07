from learning.common.paths import (
    YESNO_AUDIO_DIR,
)

import torchaudio
from pathlib import Path
from plot_set import plot_waveform, plot_waveform_fft
from torch.utils.data import Dataset

# 数据集位置
root_path = str(YESNO_AUDIO_DIR.parent)

# 下载数据集 dataset_yesno
yes_no = torchaudio.datasets.YESNO(root=root_path, download=False)
print("YESNO dataset is available locally.")

# 定义自己的数据类
class Mydataset(Dataset):
    def __init__(self, data_path, label_path):
        # 保存数据根目录和标签子目录
        self.data_path = data_path
        self.label_path = label_path
        # 拼接得到音频文件所在完整路径
        self.path = str(Path(self.data_path) / self.label_path)
        # 列出目录下所有音频文件名
        self.audiofile = sorted(Path(self.path).iterdir())

    def __getitem__(self, index):
        # 根据索引取出对应的音频文件名
        audio_path = self.audiofile[index]
        # 拼接音频文件的完整路径
        wave_path = str(audio_path)
        # 加载音频，得到波形和采样率
        waveform, sr = torchaudio.load(wave_path)
        # 返回波形、采样率和文件名
        return waveform, sr, audio_path.name

    def __len__(self):
        # 返回数据集样本总数
        return len(self.audiofile)

# 数据根目录
data_path = str(YESNO_AUDIO_DIR.parent)
# 音频文件所在的子目录名
label_path = "waves_yesno"
# 实例化自定义数据集
mydata = Mydataset(data_path=data_path, label_path=label_path)
# 指定要读取的样本索引
index = 3
# 获取一条数据：波形、采样率、文件名
waveform, sr, name = mydata[index]
# 利用plot函数进行画图分析
plot_waveform(waveform=waveform.T[:,0],sr=sr)
plot_waveform_fft(waveform=waveform.T[:,0],sr=sr,n_fft=1024)
print(name)
