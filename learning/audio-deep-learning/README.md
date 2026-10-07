# 音频深度学习与 PyTorch 练习

本目录记录 PyTorch 张量、数据集读取、固定长度/采样率预处理、Mel 频谱和 TensorBoard 可视化。当前还没有训练分类器的完整流程：`dataset_ubs8k.py` 负责产出特征和标签，其他脚本用于理解数据与模型组件。

## 建议阅读顺序

1. `tensor.py`：张量类型、形状变换、逐元素运算、矩阵乘法、拼接及 NumPy 转换。
2. `myclass.py`：Python 类、实例方法和 `__call__` 魔术方法。
3. `dataset.py`：用 TorchAudio 的 YESNO 数据和自定义 `Dataset` 读取音频。
4. `dataloader.py`：将音频统一成固定采样点数/采样率，再由 `DataLoader` 批量加载。
5. `dataset_ubs8k.py`：读取 UrbanSound8K 元数据，变为单声道、重采样、裁剪/补零，再计算 Mel 频谱。
6. `summary_writer*.py`：记录标量、音频、图片和网络结构；比较显式定义层与 `Sequential` 写法。

## 张量基础与模型结构

`tensor.py` 演示整数/浮点 Tensor、全零/随机 Tensor、`view`/`reshape` 形状变化、类型转换和逐元素算术。`torch.matmul` 是矩阵乘法；对两个一维向量执行时结果为内积标量。`torch.from_numpy` 与 NumPy 数组共享底层内存，而 `torch.tensor` 会创建副本。`resize_` 可能改变底层存储和语义，学习形状变换时优先使用 `reshape` 或 `view`（后者要求内存连续）。CPU Tensor 才能直接 `.numpy()`；GPU Tensor 需要先 `.cpu()`。

`summary_write_nn.py` 定义 3 个卷积/池化阶段，再展平为全连接层，输入假设为 `[N,3,32,32]`。padding=2 保持卷积后的宽高，三次 2 倍池化将 32×32 变为 4×4，因此全连接输入为 `64×4×4`，输出是每个样本 10 个分数。这里没有激活函数或训练步骤，只展示前向形状和 TensorBoard 模型图。`summary_writer_nn_sequential.py` 用顺序容器重写类似结构，可对照模块组织方式；当前 `MyNet(nn.Sequential)` 又把层放在 `self.module` 中并重写 forward，理解时可继续试试直接继承 `nn.Module` 或直接让 `MyNet` 继承 `nn.Sequential`。

## 音频数据集与变换

### YESNO：`dataset.py` 与 `dataloader.py`

YESNO 是随仓库保留的 60 个短语音 WAV，位置为 `data/yesno/waves_yesno/`。`dataset.py` 先确认 TorchAudio 数据目录可用，再用自定义 `Mydataset` 返回 `(waveform, sample_rate, filename)`；示例读取索引 3 并画波形/频谱。`dataloader.py` 的 `MyData` 通过 `reshape_if_necessary` 截断或在末尾补零到固定样本数，通过 `resample_if_necessary` 统一采样率，并把多声道均值合成单声道，最终由 `DataLoader` 组成 batch。

`reshape_if_necessary` 假设波形形状为 `[channel,time]`。这里的目标 8000 Hz、20000 个样本对应约 2.5 秒；处理顺序是先混声道、重采样，再截断/补零，这样最终长度仍符合设定值。

### UrbanSound8K：`dataset_ubs8k.py`

数据集默认从 `.local-data/urbansound8k/UrbanSound8K` 读取 `audio/fold*` 和 `metadata/UrbanSound8K.csv`。CSV 每行对应一个剪辑；`fold` 列定位音频目录，`classID` 为类别标签。`UrbanSound8K` 读取波形后依次混为单声道、重采样到 16 kHz、裁剪/补零到 50000 个样本（约 3.125 秒），然后计算 64 个 Mel 频带的谱图。返回值形状约为 `[1,64,time]`，标签是类别 ID。数据集并不会自动下载；下载、校验和目录设置见 `data/README.md`。

`n_fft=1024`、`hop_length=512` 控制时频分辨率，`n_mels=64` 控制 Mel 频带数。若模型训练使用数据，训练/验证拆分应按官方 fold 设计，避免随机拆分造成相邻或来源相同录音泄漏。当前脚本只取一条样本作图，没有训练或评估。

### 绘图函数

`plot_set.py` 提供波形、FFT 与谱图工具。FFT 图应关注幅值而非复数值；FFT 的 `n_fft` 小于波形长度时会截断，大于时会补零。谱图的坐标目前是矩阵索引，并非自动换算的秒和 Hz。

## TensorBoard 记录

- `summary_writer.py` 写入 `y=2x` 标量和若干 YESNO 音频示例。
- `summary_writer_img.py` 对 `assets/nn.png` 做 ToTensor、Normalize、Resize、RandomCrop，并记录图像。RandomCrop 每次可能不同。
- `summary_write_nn.py` 把网络计算图写入 `log_net/`。

运行入口将日志放在 `.local-output/audio-deep-learning/` 下，已由 Git 忽略。可在仓库根目录执行 `tensorboard --logdir .local-output/audio-deep-learning` 查看。务必关闭 `SummaryWriter`；脚本通常调用 `close()`，或在较长实验中使用上下文管理器。

## 运行与依赖

```bash
conda activate <你的环境名>
python requirements/sync_requirements.py learning/audio-deep-learning/dataset.py
python requirements/sync_requirements.py learning/audio-deep-learning/dataloader.py
```

本项目清单为 `requirements/audio-deep-learning.txt`。新环境首次使用时运行 `python -m pip install -r requirements/audio-deep-learning.txt`。PyTorch 提供 Tensor/模型/Dataset；TorchAudio 提供音频读取、重采样及 Mel 变换；TorchVision/Pillow 用于图像变换；TensorBoard 用于实验记录；librosa、NumPy、Matplotlib 用于谱图处理和绘制；pandas 用于读取 UrbanSound8K CSV。具体清单取决于生成时激活的环境。PyTorch、TorchAudio、TorchVision 要配套安装；GPU 版本按本机 CUDA/驱动选择。

YESNO 示例不需要联网。UrbanSound8K 示例需要按 `data/README.md` 准备数据；图像记录脚本需要 `assets/nn.png`。统一运行器负责仓库路径导入及依赖清单同步，不要把 `sys.path` 注入或 pip 安装代码复制到各学习脚本。
