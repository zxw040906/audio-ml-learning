# 音频深度学习与 PyTorch 练习

本目录记录 PyTorch 张量、数据集读取、固定长度/采样率预处理、Mel 频谱和 TensorBoard 可视化。当前还没有训练分类器的完整流程：`dataset_ubs8k.py` 负责产出特征和标签，其他脚本用于理解数据与模型组件。

## 学习路线与代码

1. `tensor.py`：张量类型、形状变换、逐元素运算、矩阵乘法、拼接及 NumPy 转换。
2. `myclass.py`：Python 类、实例方法和 `__call__` 魔术方法。
3. `dataset.py`：用 TorchAudio 的 YESNO 数据和自定义 `Dataset` 读取音频。
4. `dataloader.py`：将音频统一成固定采样点数/采样率，再由 `DataLoader` 批量加载。
5. `dataset_ubs8k.py`：读取 UrbanSound8K 元数据，变为单声道、重采样、裁剪/补零，再计算 Mel 频谱。
6. `summary_writer*.py`：记录标量、音频、图片和网络结构；比较显式定义层与 `Sequential` 写法。

`test.py` 只打印 PyTorch 版本，用于确认当前解释器；`plot_set.py` 是绘图辅助模块。数据流可以按 **WAV → waveform `[channel,time]` → 重采样/定长 → Mel 频谱 `[channel,mel_bin,frame]` → DataLoader/模型** 来复习。当前脚本只演示各环节，尚未连接成训练分类器。

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

`plot_set.py` 提供波形、FFT 幅值与谱图工具。FFT 的 `n_fft` 小于波形长度时会截断，大于时会补零；因此改变它不只是改变画图精度。谱图坐标目前是矩阵索引，并非自动换算的秒和 Hz。

`dataset_ubs8k.py` 产生的 `MelSpectrogram` 默认是**功率谱**；当前 `plot_set.py` 的谱图函数调用 `librosa.amplitude_to_db`，这是按幅度量转换。阅读显示出的 dB 数值时要留意这一不一致；功率谱应按功率定义转换。

## 参数影响与常见问题

| 修改项 | 影响 | 需要检查 |
| --- | --- | --- |
| `target_sr` | 重采样后的每秒样本数、频率范围 | MelSpectrogram 的 `sample_rate` 要与实际波形一致；重采样后必须再定长 |
| `target_samples` | 最终波形长度与时长 `target_samples / target_sr` | 值太小会截断声音，太大会增加补零和计算量 |
| 单声道混合 | 把多个通道平均为一个通道 | 丢失左右声道空间差异；模型输入通道数须一致 |
| `batch_size/shuffle/drop_last` | 每批样本数、顺序和末批处理 | `batch_size` 增大会占更多内存；`drop_last=True` 会丢掉不足一批的样本 |
| `dataset.py` 的 `index` | 选取哪条 YESNO 音频 | 值超出数据集长度会报索引错误；换索引后图像和文件名都随之变化 |
| `n_fft/hop_length` | 每帧频率分辨率与时间帧数量 | 更大的 `n_fft` 通常使频率刻度更细、时间定位更粗；更小的 hop 增加帧数和计算量 |
| `n_mels` | Mel 特征的频带数与输入高度 | 改动后模型输入形状可能不匹配；过多频带相对 FFT 频点可能出现空滤波器警告 |
| `Normalize` 的均值/标准差 | 图像像素缩放与颜色分布 | 通道数要匹配；标准差不能为 0；可视化归一化后的 Tensor 不等同于原图 |
| `Resize/RandomCrop` | 图像高宽及裁剪区域 | 裁剪尺寸不能超过输入；随机裁剪每次结果可能不同；全连接层输入宽度可能需要调整 |
| CNN 的输入尺寸与通道数 | 卷积输出和 `Linear.in_features` | 现有网络假设 `[N,3,32,32]`；输入改变后固定的 `64×4×4` 可能失效 |

`dataset_ubs8k.py` 的处理顺序是混单声道 → 重采样 → 定长 → Mel 变换。这个顺序保证 `target_samples` 是**目标采样率下**的样本数。`dataloader.py` 也按相同顺序处理 YESNO，便于组成等长 batch。

## TensorBoard 记录

- `summary_writer.py` 写入 `y=2x` 标量和若干 YESNO 音频示例。
- `summary_writer_img.py` 对 `assets/nn.png` 做 ToTensor、Normalize、Resize、RandomCrop，并记录图像。RandomCrop 每次可能不同。
- `summary_write_nn.py` 把网络计算图写入 `log_net/`。

运行入口将日志放在 `.local-output/audio-deep-learning/` 下，已由 Git 忽略。可在仓库根目录执行 `tensorboard --logdir .local-output/audio-deep-learning` 查看。务必关闭 `SummaryWriter`；脚本通常调用 `close()`，或在较长实验中使用上下文管理器。

当前 `summary_writer.py` 在写入第 8 条音频时调用 `exit()`，因此后面的 `file1.close()` 不会执行。观察音频日志时若发现最后一条缺失，需考虑写入器尚未正常关闭；这属于现有脚本的实现限制。

## 运行与依赖

```bash
conda activate <你的环境名>
python learning/audio-deep-learning/dataset.py
python learning/audio-deep-learning/dataloader.py
```

本项目清单为 `requirements/audio-deep-learning.txt`。新环境首次使用时运行 `python -m pip install -r requirements/audio-deep-learning.txt`。PyTorch 提供 Tensor/模型/Dataset；TorchAudio 提供音频读取、重采样及 Mel 变换；TorchVision/Pillow 用于图像变换；TensorBoard 用于实验记录；librosa、NumPy、Matplotlib 用于谱图处理和绘制；pandas 用于读取 UrbanSound8K CSV。具体清单取决于生成时激活的环境。PyTorch、TorchAudio、TorchVision 要配套安装；GPU 版本按本机 CUDA/驱动选择。

上面命令从仓库根目录运行，会直接使用当前激活的 Python。直接运行不会同步依赖清单；脚本启动时会先把仓库根目录加入模块搜索路径，让 `from learning.common.paths import ...` 可用，也会加入脚本目录以便导入同目录的 `plot_set.py`。所以在仓库根目录可用相对路径运行，也能从任意工作目录使用绝对路径运行，例如：

```bash
/home/zxw/anaconda3/envs/pytorch/bin/python \
  /home/zxw/python_code/project/audio-ml-learning/learning/audio-deep-learning/dataloader.py
```

如果希望运行前同步依赖清单，则从仓库根目录使用统一入口：

```bash
python requirements/sync_requirements.py learning/audio-deep-learning/dataloader.py
```

两种方式都使用当前选定的 Python 环境。YESNO 示例不需要联网。UrbanSound8K 示例需要按 `data/README.md` 准备数据；图像记录脚本需要 `assets/nn.png`。

## 复习检查

能否从 `target_samples` 和 `target_sr` 算出音频时长？为什么定长必须放在重采样后？把 `n_mels` 从 64 改成 128 后，哪一个张量维度会变化，后续模型要检查什么？
