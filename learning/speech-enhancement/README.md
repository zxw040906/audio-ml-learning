# 单通道语音增强练习

本目录用合成白噪声构造带噪语音，以干净语音为监督目标，练习 `Dataset`、一维卷积、训练/验证和 SNR 评估。它是单通道时域语音增强练习；它不使用麦克风阵列或空间位置标签，因此不能直接作为区域拾音算法。区域拾音规划仍见 [`projects/area-pickup/README.md`](../../projects/area-pickup/README.md)。

## 流程与文件

| 文件 | 用途 |
| --- | --- |
| `mix_audio.py` | 按目标 SNR 缩放白噪声并与语音相加 |
| `dataset.py` | 延迟读取 FLAC、统一长度、生成可复现噪声，返回 `(mixture, clean)` |
| `model.py` | 定义基础 CNN、空洞 CNN 和残差去噪 CNN |
| `train.py` | 按说话人划分训练/验证集，用 MSE 和 Adam 训练残差 CNN |
| `evaluate.py` | 加载残差模型，计算单条验证样本 SNR 并保存试听音频 |
| `evaluate_batch.py` | 汇总多条样本的 SNR 改善；需要三种模型权重均已生成 |
| `analyze_spectrum.py` | 比较 `output/` 中预先生成的评估音频频谱 |

张量输入输出形状为 `[B, T]`，模型内部扩展为 `[B, 1, T]` 以适配 `Conv1d`，最终回到 `[B, T]`。默认采样率为 16 kHz、裁剪/补零长度为 2 秒（32000 点）、混合信噪比为 5 dB。`train.py` 取最多 40 条训练语音和 10 条验证语音，batch size 为 4，训练 5 个 epoch，使用 CPU、MSELoss 和 Adam（学习率 0.001）。这些是教学用的小规模设置，不代表充分训练或对真实噪声的泛化结果。

## 准备本地数据

脚本默认从本目录的 `data/LibriSpeech/dev-clean/` 递归查找 FLAC，数据不纳入 Git。请从 [OpenSLR LibriSpeech 页面](https://www.openslr.org/12) 获取合法的 `dev-clean` 数据包，将其解压到上述目录，并遵守数据集随附的许可与使用说明。需要至少两个说话人，否则 `split_by_speaker()` 无法划分训练集和验证集。

目录结构示例：

```text
learning/speech-enhancement/data/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac
```

当前数据集实现要求音频为 16 kHz 单声道；遇到其他采样率或多声道文件会报错，不会自动重采样或混音。读取的音频不足 2 秒时补零，超过 2 秒时截断。

## 运行

在仓库根目录执行。请使用已安装 PyTorch、NumPy、SoundFile 和 Matplotlib 的 Python 环境：

```bash
python learning/speech-enhancement/dataset.py
python learning/speech-enhancement/train.py
python learning/speech-enhancement/evaluate.py
```

训练后会在本地 `output/best_residual_model.pth` 写入最佳验证损失对应的权重；评估脚本写入 clean、mixture 和 predicted WAV。`evaluate_batch.py` 还需要 `best_model.pth` 与 `best_dilated_model.pth`，但当前 `train.py` 只训练并保存残差 CNN，所以这两个权重须由对应模型的训练实验单独生成。`analyze_spectrum.py` 读取 `eval_clean.wav`、`eval_mixture.wav`、`eval_predicted.wav`；这些文件需由其他评估流程生成，当前 `evaluate.py` 输出的是 `residual_*.wav`，名称并不匹配。

## 实现解读与注意点

### 合成噪声和目标 SNR

若干净语音功率为 \(P_s\)，白噪声功率为 \(P_n\)，目标信噪比为 \(S\) dB，`add_noise()` 将噪声乘以

\[
\alpha = \sqrt{\frac{P_s}{P_n\,10^{S/10}}}, \qquad y = s + \alpha n.
\]

因此较低的目标 SNR 表示更强噪声。`dataset.py` 以 `42 + index` 为随机种子生成每个样本的噪声，便于复现；改变样本顺序或索引可能改变噪声。训练集和验证集按说话人划分，以避免同一说话人同时出现在两侧；但样本数还受到 `max_files` 截断，实验结论只适用于该小样本设置。

### 三种卷积结构

- `SimpleDenoisingCNN`：三层普通卷积直接回归干净波形。
- `DilatedDenoisingCNN`：膨胀率依次为 1、2、4、8、16，在不增加池化降采样的情况下扩大时间感受野。
- `ResidualDenoisingCNN`：预测噪声并计算 `mixture - predicted_noise`；末层权重和偏置初始化为零，因此初始化时输出等于输入。

当前 `train.py` 只选择残差 CNN，损失是整段波形的均方误差。`evaluate.py` 的 SNR 用参考干净语音与误差计算；`evaluate_batch.py` 报告逐条 SNR 改善。SNR 是波形误差指标，不能单独表示主观听感、语音可懂度或空间拾音效果。

## 本地文件与版本控制

`data/` 中的数据集副本和 `output/` 中的权重、音频及图表都由 `.gitignore` 排除。可提交代码和本 README；提交前用 `git status --short` 检查，并确认这些本地文件没有被此前跟踪。其他学习项目的数据和输出规则见仓库根目录的 [`data/README.md`](../../data/README.md)。
