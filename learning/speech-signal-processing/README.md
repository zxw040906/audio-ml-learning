# 语音信号处理练习

本目录从离散音频波形出发，练习时域特征、短时傅里叶变换（STFT）和频谱特征。音频都是本地素材，默认位于 `assets/audio/`，脚本读取时通过 `learning/common/paths.py` 定位。

## 学习路线与代码

按 **采样率与波形 → 分帧 → 幅值包络/RMS/ZCR → FFT/STFT → 谱质心** 的顺序阅读。时域特征回答“某段时间的振幅和变化如何”，频域特征回答“哪些频率成分更强”，STFT 则把频率分析放回连续的时间帧上。手写版本帮助理解步骤，librosa 版本用于对照边界与默认参数。

## 数据约定

`bird.wav`、`time.wav`、`water.wav` 是示例音频。使用 `librosa.load(..., sr=None)` 保留文件原始采样率；省略 `sr=None` 时 librosa 会默认重采样到 22050 Hz。采样率决定样本索引和时间/频率刻度的换算，画图或比较音频前要确认各信号的采样率一致。部分示例只读取前 5 秒。

## 时域特征

### 幅值包络：`Time_Process/Amplitude_Envelope.py`

把信号切成重叠帧，每帧取最大幅值，得到随时间变化的粗略响度轮廓。`frame_length=1024` 是每帧样本数，`hop_length=512` 意味着帧之间重叠 50%。脚本分别手工切片和用 `librosa.util.frame` 计算，再将结果叠加在原波形上比较。当前手写实现取的是最大值而非绝对值最大值，因此对负向占优的帧不对称；分析双极性波形时常用 `max(abs(frame))`。

手写分帧使用 `np.pad(..., 'wrap')` 循环填充尾部，虽然注释写“补零”，实际并非零填充；这会影响末尾帧的包络。

### 均方根能量：`Time_Process/RMS.py`

每帧计算 `sqrt(mean(x²))`，对正负振幅都敏感，衡量局部平均能量。脚本将手写实现与 `librosa.feature.rms` 对比。帧长 1024、帧移 512；以采样率 `sr` 计算，帧长对应 `1024/sr` 秒，不能把注释里的 48 kHz 当成所有素材的采样率。

手写分帧没有中心补边，而 `librosa.feature.rms` 默认进行中心对齐；首尾帧和帧数可能不同，比较两条曲线前应统一对齐规则。

### 过零率：`Time_Process/ZCR.py`

统计帧内信号符号改变的频繁程度。高频、噪声或清辅音通常过零较多，但 ZCR 不能单独判断音类别。实现使用符号差分并除以 `2 × frame_length`；与 librosa 比较时要留意边界帧策略、零值符号和归一化定义不同，因此曲线可能略有偏差。

## 频域与时频分析

### 手写 STFT：`Frequency_Process/STFT.py`

主要步骤是补零、分帧、乘 Hann 窗、逐帧做实数 FFT，再把幅度平方换算为功率 dB。`frame_size=1024` 控制频率分辨率，`hop_size=512` 控制时间采样密度；较长帧通常带来更细频率分辨率但时间定位变粗。实数 FFT 只保留非负频率，频点 `k` 对应 `k × sr / n_fft` Hz。功率转 dB 使用 `10 log10(P)`，幅度转 dB 使用 `20 log10(A)`。绘图纵轴刻度由频点换算为 Hz。

当前数组形状为 `[时间帧, 频率 bin]`，直接传给 `imshow` 时横轴是频率、纵轴是时间；代码却把纵轴刻度标成 Hz。复习时以数组形状为准，要让频率在纵轴需转置数组并同步改坐标。

### librosa STFT：`Frequency_Process/STFT_librosa.py`

调用 `librosa.stft` 计算时频复数矩阵，随后展示波形、FFT 和幅度谱图。矩阵的行对应频率 bin、列对应时间帧；`abs` 取模长后丢弃相位。与手写版本对照时还应比较窗函数、中心补零、边界处理和幅值归一化，不能只看图像是否相似。

### 谱质心与带宽：`Frequency_Process/SC_BW.py`

脚本计算了三段音频的 spectral centroid（谱质心），它是频谱能量的加权平均频率，可粗略描述“明亮度”。变量 `sc_bird/sc_time/sc_water` 已计算，但现有绘图展示的是三段波形，并未绘出质心或谱带宽。后续可将 `librosa.feature.spectral_bandwidth` 加入，并把质心按帧时间叠加到波形或谱图上；本脚本当前名称中的 BW 尚未对应实际带宽计算。

代码连续读取三段文件时复用了变量 `sr`，计算全部谱质心时用的是最后一段的采样率；若文件采样率不同，前两段的频率值会有误，应分别保存各自的采样率。

### 通用绘图：`Frequency_Process/plot_set.py`

提供波形、实数 FFT **幅值**和谱图绘制函数。FFT 长度受 `n_fft` 控制：小于原波形长度时会截断，大于时会补零。谱图以幅度转 dB，坐标轴目前是帧数和频率 bin 的索引；若需要秒/Hz，应按 hop length 与采样率生成坐标。

## 参数影响与常见问题

| 修改项 | 影响 | 需要检查 |
| --- | --- | --- |
| `sr=None` 或指定采样率 | 是否保留原始采样率，或在读入时重采样 | 多段音频比较前确认采样率；频率 bin 换算要使用实际 `sr` |
| `duration` | 读取音频的时间范围 | 只读前 5 秒时，后续事件不会进入特征或图像 |
| `frame_length/frame_size` | 每帧时间长度约为 `frame_length / sr` | 帧越长频率分辨率通常越细、时间定位越粗；音频短于帧长时分帧可能失败 |
| `hop_length/hop_size` | 相邻帧的步进与重叠 | hop 变小会增加帧数和计算量；改变后绘图时间轴须重算 |
| `n_fft` | FFT 频点数与频率间隔 `sr / n_fft` | `n_fft` 与窗长不同会引入截断或补零；画图频率刻度必须同步更新 |
| 窗函数、补零、`center` | 频谱泄漏和首尾帧对齐 | 手写 STFT 与 librosa 默认边界处理不同，图像和帧数未必一致 |
| 幅度/功率转 dB | 对数缩放后的数值与显示范围 | 幅度用 `20 log10`，功率用 `10 log10`；零值需避免 `log10(0)` 得到负无穷 |
| 过零率边界与零值规则 | 每帧 ZCR | 自写符号差分与 librosa 的中心填充、零值处理可能不同 |

手写幅值包络目前取每帧的**最大原始值**，不是绝对峰值，负向较强的帧会被低估。`RMS.py` 与 `ZCR.py` 注释中的 48 kHz 只是换算示例，真实帧时长须用文件实际采样率。`SC_BW.py` 计算谱质心但只画波形，尚未计算谱带宽；不要把标题当作已实现功能。

## 入口脚本

根目录 `test.py` 是幅值包络的早期对照版本，和 `Time_Process/Amplitude_Envelope.py` 内容基本重复；推荐以分主题目录为主。`test.py` 与各计算脚本运行后会打开 Matplotlib 窗口，需要本地素材可读。

## 运行与依赖

在仓库根目录激活已有 Conda/Python 环境后运行：

```bash
conda activate <你的环境名>
python requirements/sync_requirements.py learning/speech-signal-processing/Time_Process/RMS.py
python requirements/sync_requirements.py learning/speech-signal-processing/Frequency_Process/STFT_librosa.py
```

依赖清单为 `requirements/speech-signal-processing.txt`。新环境首次使用时运行 `python -m pip install -r requirements/speech-signal-processing.txt`。librosa 用于读取音频、分帧和特征/STFT；NumPy 用于数组计算；Matplotlib 用于可视化。无需 UrbanSound8K 数据集。统一运行入口负责仓库路径导入，依赖同步只在 `requirements/sync_requirements.py` 维护。

## 复习检查

给定 `sr=16000`、帧长 1024、帧移 512，能否算出一帧和一帧移分别对应多少毫秒？手写 STFT 与 librosa STFT 为何可能帧数不同？谱质心能说明什么，为什么单看波形还不能验证它？
