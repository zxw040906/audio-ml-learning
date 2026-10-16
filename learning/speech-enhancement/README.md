# 单通道语音增强练习

本目录用合成白噪声构造带噪语音，以干净语音为监督目标，练习 `Dataset`、一维卷积、训练/验证和 SNR 评估。它提供从单通道信号建模走向空间音频问题的基础，但不使用麦克风阵列或空间位置标签，尚未实现区域拾音算法。区域拾音规划仍见 [`projects/area-pickup/README.md`](../../projects/area-pickup/README.md)。

## 流程与文件

| 文件 | 用途 |
| --- | --- |
| `mix_audio.py` | 按目标 SNR 缩放白噪声并与语音相加 |
| `dataset.py` | 延迟读取 FLAC、统一长度、生成可复现噪声，返回 `(mixture, clean)` |
| `model.py` | 定义基础 CNN、空洞 CNN 和残差去噪 CNN |
| `train.py` | 按说话人划分训练/验证集，用 MSE 和 Adam 训练残差 CNN |
| `evaluate.py` | 加载残差模型，计算单条验证样本 SNR 并保存试听音频 |
| `evaluate_batch.py` | 汇总多条样本的 SNR 改善；需要三种模型权重均已生成 |
| `evaluate_test.py` | 在 `test-clean` 上按说话人抽样，比较固定模型在两种噪声下的指标 |
| `evaluate_noise_type.py` | 用同一残差模型和固定验证文件比较白噪声、低通噪声 |
| `evaluate_frequency_error.py` | 比较两种噪声条件下的四频段误差、噪声能量及频带相关性 |
| `audio_utils.py` | 统一导出 SNR、整体/频带 MSE、频带能量、归一化内积和按说话人抽样工具 |
| `analyze_spectrum.py` | 比较 `output/` 中预先生成的评估音频频谱；当前输入命名与 `evaluate.py` 输出不一致 |

张量输入输出形状为 `[B,T]`，模型内部扩展为 `[B,1,T]` 以适配 `Conv1d`，最终回到 `[B,T]`。默认采样率为 16 kHz、裁剪/补零长度为 2 秒（32000 点）、混合信噪比为 5 dB。`train.py` 先按说话人划分，再分别截取最多 40 条训练语音和 10 条验证语音；batch size 为 4，训练 5 个 epoch，使用 CPU、`MSELoss(reduction='mean')` 和 Adam（学习率 0.001）。训练脚本当前只训练 `ResidualDenoisingCNN`，并按最低验证损失保存权重。上述是教学用小规模设置，不代表充分训练或对真实噪声的泛化结果。

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

训练后会在本地 `output/best_residual_model.pth` 写入最低验证损失对应的权重。`evaluate.py` 载入该权重，对验证集第一条样本计算输入/输出 SNR，并写入 `residual_clean.wav`、`residual_mixture.wav`、`residual_predicted.wav`。`evaluate_batch.py` 会在同一验证集上逐条计算三种模型指标，但它还需要 `best_model.pth` 与 `best_dilated_model.pth`；当前 `train.py` 并未训练或保存这两种模型，因此完整三模型比较流程尚不能只靠现有训练入口完成。`analyze_spectrum.py` 读取 `eval_clean.wav`、`eval_mixture.wav`、`eval_predicted.wav`，与 `evaluate.py` 的 `residual_*.wav` 命名不匹配；需由其他流程准备其输入，当前脚本之间不能直接串接。

## 实现解读与注意点

### 物理直觉：加性噪声与功率

单通道录音可用加性模型表示为

$$
y[t]=s[t]+n[t]
$$

其中 $s[t]$ 是干净语音，$n[t]$ 是噪声，$y[t]$ 是混合波形；下标 $t$ 表示离散样本索引，样本幅值在本实验中为无量纲浮点数。模型假设噪声与语音按样本相加，未描述混响、非线性削波、时变麦克风响应或阵列空间效应。白噪声在生成模型中是各样本独立同分布的随机噪声；有限长度样本的经验均值和方差不保证恰好为理论值。

对有 $T$ 个样本的序列 $x[t]$，平均平方功率定义为

$$
P_x=\frac{1}{T}\sum_{t=0}^{T-1}x[t]^2
$$

它是同一幅值标度下比较能量强弱的功率代理；若是电压且负载相同，物理平均功率与均方电压成正比。由于 `add_noise()` 对等长序列分别取 `np.mean(x**2)`，变量 `clean_power` 和 `noise_power` 就对应这个定义。

### 低通随机噪声与频谱分析

`noise_type.py::generate_low_frequency_noise(length, a=0.95, seed=42)` 使用 `np.random.default_rng(seed)` 创建局部随机数生成器，再由 `standard_normal(length)` 生成长度为 `length` 的标准正态白噪声序列 $w[n]$。逐样本递推

$$
v[n]=a v[n-1]+(1-a)w[n]
$$

其中实现的初始值为 $v[0]=(1-a)w[0]$。$a=0.95$ 使前一采样状态保留较多、每一步新噪声输入权重较小，因此生成的合成一阶低通随机噪声更平滑、能量更集中于低频；同一长度和种子会生成可复现序列。它是合成噪声模型，并不等同于真实汽车发动机噪声。

频谱分析步骤用 `np.fft.rfft()` 计算实信号单边频谱，`np.fft.rfftfreq()` 生成频率坐标，`np.abs()` 取幅度后平方得到各频点功率，再以 `np.sum()` 汇总指定频带和全频带功率。频率分辨率为 $f_s/N$，其中 $f_s$ 是采样率、$N$ 是信号长度。偶数长度实信号的 `rfft` 结果包含 DC 和 Nyquist 端点，它们不加倍；其余正频率功率需乘 2，以补上共轭对称的负频率部分（代码形式为 `power[1:-1] *= 2`）。奇数长度没有 Nyquist 独立端点，除 DC 外均需乘 2（`power[1:] *= 2`）。计算低频能量占比时，分子是截止频率内修正后的功率和，分母是所有频点修正后的功率和；若分子和分母使用相同 FFT 标度，公共因子会相消。`analyze_spectrum.py::calculate_low_frequency_ratio` 实现了这一计算。

已有实验记录中，白噪声和一阶低通合成噪声使用 16 kHz 采样率、长度 16,000、随机种子 42；干净语音一项为 10 条验证语音低频占比的平均值。结果如下，均为历史记录而非本次重新运行：

| 信号 | 0–500 Hz 能量占比 |
| --- | ---: |
| 白噪声 | 6.23% |
| 一阶低通合成噪声 | 84.30% |
| 10 条干净验证语音平均值 | 62.33% |

低频噪声由 $v[n]=0.95v[n-1]+0.05w[n]$ 生成，其中 $w[n]$ 是标准正态白噪声，初始状态按实现取 $v[0]=0.05w[0]$。语音在 0–500 Hz 也有显著能量，所以语音与低频噪声的频谱存在重叠。这种重叠提供了分析降噪难度的背景，但单凭频谱重叠不能证明模型降噪失败的原因；仍需结合误差及噪声估计结果判断。

### 从目标 SNR 推导噪声缩放

目标信噪比 $S$ dB 定义为干净信号功率与缩放后噪声功率的比值：

$$
S=10\log_{10}\left(\frac{P_s}{P_{\alpha n}}\right)
$$

先将分贝式两侧除以 10 并取 10 次幂：

$$
10^{S/10}=\frac{P_s}{P_{\alpha n}}
$$

若缩放系数 $\alpha$ 是实数，缩放后功率为

$$
P_{\alpha n}=\frac{1}{T}\sum_t(\alpha n[t])^2=\alpha^2\frac{1}{T}\sum_tn[t]^2=\alpha^2P_n
$$

代回目标比值并解 $\alpha$：

$$
10^{S/10}=\frac{P_s}{\alpha^2P_n}
\quad\Longrightarrow\quad
\alpha^2=\frac{P_s}{P_n10^{S/10}}
\quad\Longrightarrow\quad
\alpha=\sqrt{\frac{P_s}{P_n10^{S/10}}}
$$

取正根是因为只调整噪声幅度，不翻转其符号。最终混合波形为 $y=s+\alpha n$。`mix_audio.py::add_noise` 逐项实现上述运算；输入必须是一维、等长且功率为正。手算检查：若 $P_s=1$、$P_n=4$、目标 $S=0$ dB，则 $\alpha=\sqrt{1/4}=0.5$，缩放噪声功率变为 1，信噪比为 0 dB。目标 SNR 越低，分母中的 $10^{S/10}$ 越小，噪声缩放越大。

兼容练习入口 `noise_type.py::scale_noise_to_snr(clean, noise, snr_db)` 保留原调用方式，但内部委托给 `mix_audio.py::add_noise()` 取得同一套缩放结果；正式 Dataset 路径只调用 `add_noise()` 一次，由它负责目标 SNR 缩放和混合。白噪声仍由 Dataset 使用原来的局部随机数生成器生成，低频噪声仍由 `generate_low_frequency_noise()` 生成。

### MSE 目标与 SNR 指标

训练代码使用波形均方误差。对 $T$ 个样本，模型输出 $\hat s[t]$，干净目标为 $s[t]$，则

$$
L_{MSE}=\frac{1}{T}\sum_{t=0}^{T-1}(\hat{s}[t]-s[t])^2
$$

前提是预测与目标对齐且同长度。它惩罚每个采样点的平方误差；`train.py` 使用 `nn.MSELoss()` 默认 mean reduction，对 batch 和时间元素求均值。

评价脚本将误差 $e[t]=\hat s[t]-s[t]$ 视为残留噪声，并计算

$$
\mathrm{SNR}_{\mathrm{out}}=10\log_{10}\left(\frac{\sum_t s[t]^2}{\sum_t e[t]^2}\right)
$$

令 $P_s=T^{-1}\sum_ts[t]^2$、$P_e=T^{-1}\sum_te[t]^2$，则分子分母同除以 $T$，比值不变，故也等于 $10\log_{10}(P_s/P_e)$。单条和批量评估统一调用 `audio_utils.py::calculate_snr()`；输入转为 `float64`，批量输入形状为 `[B,T]` 并沿时间轴逐条计算。误差为零时返回正无穷，静音参考且误差也为零时返回 NaN，静音参考但误差非零时返回负无穷，不用 epsilon 改写指标定义。`calculate_snr_improvement()` 返回输出 SNR 减输入 SNR；逐样本 dB 的算术平均仍与先合并样本功率再计算不同。

MSE 与输出 SNR 的分子分母角色不同：对固定目标能量的同一条样本，降低误差 MSE（或误差平方和）会提高输出 SNR；但跨样本时目标能量不同，单看 MSE 不足以排序 SNR。训练还使用批次/样本均值，而脚本计算每条记录的能量比。SNR 提升定义为输出 SNR 减输入混合 SNR，单位 dB；它是参考波形误差指标，不等价于主观听感、语音可懂度或空间拾音效果。

当同一条语音的误差功率减半时，SNR 变化为 $10\log_{10}(2)\approx3.01$ dB。改善样本数按未四舍五入的逐条 `snr_improvement > 0` 统计；打印为 `-0.00 dB` 的值不能据显示文本判断符号。改善比例是改善样本数除以样本总数，它与平均 SNR 改善量回答不同问题，应一并报告。对多种噪声的统计可用外层遍历噪声类型、内层遍历语音的结构；每种噪声的平均值和改善比例必须在该噪声对应的内层循环结束后计算。

### Conv1d：计算、形状、参数与感受野

模型调用前把 `[B,T]` 变为 `[B,1,T]`，其中 $B$ 是批大小、1 是单声道通道、$T$ 是时间采样点；逐层卷积后再 squeeze 回 `[B,T]`。`model.py` 的各卷积均为 `kernel_size=9`、stride 默认 1、bias 默认启用。dilation 为 $d$ 时，有效核宽为 $k_{eff}=d(k-1)+1$，一维输出长度为

$$
T_{\mathrm{out}}=\left\lfloor\frac{T+2p-d(k-1)-1}{s}+1\right\rfloor
$$

其中 $p$ 是左右各自的 padding，$s$ 是 stride。代码对每层使用 $p=d(k-1)/2$，本模型的核宽为奇数且 stride=1，因此长度保持为 $T$。

单个输出位置、输出通道 $o$ 的计算是对所有输入通道和核位置加权求和再加偏置：

$$
z_{o,t}=b_o+\sum_{c=0}^{C_{\mathrm{in}}-1}\sum_{q=0}^{k-1}w_{o,c,q}\,x_{c,ts-p+qd}
$$

该索引形式描述 stride 与 dilation 的采样位置；越界位置由 padding 零补齐。对 `SimpleDenoisingCNN`，第一层第一个窗口是输入 9 点与 9 个卷积权重逐项乘积求和，加一个 bias；ReLU 再逐点执行 $\max(0,z)$。

普通 Conv1d 的参数量是二维卷积公式在核高为 1 时的特例：

$$
N_{\mathrm{params}}=C_{\mathrm{out}}(C_{\mathrm{in}}k+b),
$$

其中 $C_{\mathrm{in}},C_{\mathrm{out}}$ 是输入和输出通道数，$k$ 是核宽（采样点），$b=1$ 对应 `bias=True`、$b=0$ 对应 `bias=False`；此处为 `groups=1`。核参数在时间位置间共享，因此参数数不乘 $T$。原始/残差网络三层参数分别是 $16(1·9+1)=160$、$16(16·9+1)=2320$、$1(16·9+1)=145$，总计 2625。残差模型只是把最后层初值设为零，不改变参数总数。空洞网络五层参数分别为 160、2320、2320、2320、145，总计 7265；dilation 和 padding 不改变权重张量尺寸。

感受野按层计算。设初值 $r_0=1$ 个输入样本、jump $j_0=1$，第 $l$ 层 stride 为 $s_l$、核宽 $k_l$、dilation 为 $d_l$，则

$$
j_l=j_{l-1}s_l,\qquad r_l=r_{l-1}+(k_l-1)d_lj_{l-1}
$$

前提是串接的规则一维卷积；jump 表示相邻输出点在原输入上的间隔，感受野是一个输出可依赖的输入点跨度。普通三层网络每层 $k=9,d=1,s=1$：$r_1=1+8=9$，$r_2=9+8=17$，$r_3=17+8=25$ 个采样点，即 16 kHz 下跨度约 25/16000=1.5625 ms。

空洞网络按代码 dilation `[1,2,4,8,16]` 逐层计算：$r_1=1+8·1=9$，$r_2=9+8·2=25$，$r_3=25+8·4=57$，$r_4=57+8·8=121$，$r_5=121+8·16=249$ 个点；约为 15.56 ms。所有 stride=1，因此 jump 始终为 1。膨胀卷积在核元素之间留间隔，不增加参数数量或降采样，但扩大可见上下文；理论感受野不代表模型一定有效利用全部上下文。

### 普通、空洞与残差结构

| 模型 | `model.py` 结构 | 参数量 | 末层含义 |
| --- | --- | ---: | --- |
| `SimpleDenoisingCNN` | `Conv1d(1,16,9)` → ReLU → `Conv1d(16,16,9)` → ReLU → `Conv1d(16,1,9)` | 2625 | 直接输出干净波形估计 |
| `DilatedDenoisingCNN` | 5 层通道 `1→16→16→16→16→1`，核宽 9，dilation `1,2,4,8,16`，前四层后 ReLU | 7265 | 直接输出干净波形估计，理论感受野 249 点 |
| `ResidualDenoisingCNN` | 与三层普通 CNN 同样的层宽/核宽，并把末层 weight、bias 初始化为 0 | 2625 | 输出输入减去预测噪声 |

三种结构都没有池化或时间降采样，输入输出长度一致。普通 CNN 感受野为 25 点，空洞 CNN 为 249 点；残差 CNN 的卷积支路感受野仍为 25 点。模型结构/参数量来自上述卷积配置，权重初始化和训练后的数值会变，但计数不变。

### 残差去噪与零初始化

令混合输入 $y=s+n$，网络支路估计噪声 $\hat n=f_\theta(y)$，最终干净估计为

$$
\hat{s}=y-\hat{n}
$$

若初始化末层 weight 与 bias 都为零，则任意支路特征经末层线性映射后 $\hat n=0$，于是初始化时 $\hat s=y$，即模型从恒等映射开始。训练仍通过 `MSELoss(\hat s,s)` 的误差更新网络；对减法关系，$\partial\hat s/\partial\hat n=-1$，故梯度会沿预测噪声支路反传。零初始化只保证最终层输出初始为零，不表示中间层权重为零，也不保证训练后必然去噪成功。实现位于 `ResidualDenoisingCNN.__init__` 与 `forward`。

### 划分训练集、验证集与独立测试集

训练集用于更新模型参数，验证集用于选择模型和调整超参数，测试集用于模型固定后的独立性能评估。验证集即使不参与反向传播，只要参与了模型选择，就不能当作完全独立的最终测试。

`train.py::split_by_speaker` 从 `speaker/chapter/*.flac` 路径取说话人 ID，固定 NumPy 种子 42 打乱说话人，约 20% 说话人放入验证集且至少 1 个；余者用于训练。按说话人隔离可减少同一说话人音色同时进入训练与验证造成的泄漏。随后两个 Dataset 各自按排序后的路径最多取 40/10 条；这不是完整数据集训练。白噪声则由样本 index 决定种子。

每个 epoch 训练集 loader 打乱，验证集不打乱。训练步骤是 forward → batch mean MSE → `zero_grad()` → `backward()` → Adam `step()`；验证用 `model.eval()` 和 `no_grad()`，不更新权重。脚本在 5 个 epoch 中保存验证 MSE 最低时的 `state_dict` 到 `output/best_residual_model.pth`。验证集参与了权重选择，因此本次另用 LibriSpeech `test-clean` 做固定模型后的独立测试；独立划分不代表小样本结果可以代表整个测试集。训练入口仍只训练残差模型，没有在入口中比较三种架构。

### 使用 `Path` 搜索文件并按说话人抽样

`Path(__file__).resolve().parent` 取得当前脚本所在目录；`Path.rglob("*.flac")` 递归搜索所有子目录，适合 LibriSpeech 的说话人/章节层级。`glob("*.flac")` 匹配当前目录中的文件；模式包含 `**` 时也可以递归，`rglob("*.flac")` 是递归搜索的简写。`Path.relative_to(test_dir)` 可取得相对于测试集根目录的路径，例如从 `test-clean/1089/134686/file.flac` 读取说话人 ID `1089`。

`audio_utils.py::sample_files_by_speaker(test_files, test_dir, num_speakers=10, seed=42)` 使用 `defaultdict(list)` 按说话人分组，`set()` 可用于检查抽样结果中的说话人数量。函数将说话人 ID 排序后交给独立的 `random.Random(seed)` 随机数生成器：`rng.sample()` 不放回抽取不同说话人，再对每位说话人的文件列表调用 `rng.choice()` 选一条。它不消耗 NumPy 噪声生成器的随机状态；当可用说话人少于目标数时，函数使用全部可用说话人。当前实现实际使用的是 `choice()`，不是 `choices()`。

本次调试中曾写成：

```python
selected_file = rng.choices(files)
```

`choices()` 的返回值始终是列表，即使只抽一项也是列表。随后对该对象调用 `path.relative_to(test_dir)` 会得到 `AttributeError: 'list' object has no attribute 'relative_to'`。需要单个 `Path` 时应使用：

```python
selected_file = rng.choice(files)
```

另外两处容易覆盖或忽略分层抽样结果：抽样后重新赋值 `selected_files = test_files[:10]` 会覆盖选中结果；构造 `SpeechEnhancementDataset` 时传入 `files=test_files` 也会绕过抽样。应始终将 `selected_files` 传给 Dataset。`evaluate_test.py` 在噪声类型循环外生成一次这份列表，然后在 `white` 与 `low_frequency` 两种条件下复用，确保比较使用相同的干净语音。

### Dataset 噪声类型与数据流

`dataset.py::SpeechEnhancementDataset` 默认 16 kHz、2.0 秒（32000 点）、5 dB、最多 20 文件；训练入口分别以 40 和 10 的 `max_files` 取训练/验证文件。每个索引使用 `default_rng(42 + index)` 生成噪声，因此同一数据集索引可复现；样本排序、文件集合或索引变化会改变对应噪声。补零/截断发生在生成噪声之前，故二者长度一致。

数据集 `SpeechEnhancementDataset` 的 `noise_type` 默认值是 `"white"`，保留原有白噪声生成方式；`"low_frequency"` 分支使用 `a=0.95`、`seed=42+index`。过去曾在 `dataset.py` 中导入 `evaluate_batch.calculate_snr`，而 `evaluate_batch.py` 同时导入 `dataset.SpeechEnhancementDataset`，形成循环依赖：加载其中一个模块时又回头请求尚未完成初始化的另一个模块。当前修复是在 `dataset.py` 的演示入口中直接对该条样本计算输入 SNR，不再导入 `evaluate_batch`；批量评估脚本仍由 `evaluate_batch.py` 提供，Dataset 的正式混合仍只走 `add_noise()`。

## 实验结果与分析

### 三种 CNN 的历史验证集比较

以下数值来自已有学习记录：在 10 条固定 5 dB 白噪声验证样本上的历史三模型比较，**不是本次或当前仓库重新运行的结果**。验证集参与模型选择，因此这些结果只能作为该学习实验的验证记录，不能视为独立测试表现。

| 模型 | 参数量 | 平均 SNR 提升 | 最小提升 | 最大提升 | 改善样本数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 CNN | 2625 | +1.64 dB | -1.26 dB | +4.98 dB | 8/10 |
| 空洞 CNN | 7265 | -0.39 dB | -2.19 dB | +2.07 dB | 4/10 |
| 残差 CNN | 2625 | +3.13 dB | +1.02 dB | +5.55 dB | 10/10 |

`evaluate.py` 对固定的一条验证样本报告输入 SNR、输出 SNR 和差值；`evaluate_batch.py` 对每条分别算能量比，再对 10 个 dB 改善值做算术平均，并给出最小、最大及改善数。平均“逐样本 dB”不同于先汇总全部样本信号能量再取一次对数；后者会被能量更大的样本主导。两种统计口径回答的问题不同，表格采用 `evaluate_batch.py` 的逐条 SNR 提升算术平均口径。少量语音、单一白噪声类型、固定 5 dB、说话人数量及截断长度都限制了外推能力。

### 固定输入 SNR 下的噪声类型比较

**实验记录：2026-10-11（W03-02）。**

`evaluate_noise_type.py` 从 `train.py::split_by_speaker()` 取得同一验证文件列表，加载固定的 `output/best_residual_model.pth` 权重，并分别创建最多 10 条、16 kHz、2 秒、目标输入 SNR 5 dB 的 Dataset；同一个已加载的 `ResidualDenoisingCNN` 依次评估白噪声和低通噪声，只改变噪声类型，不重新训练模型。指标由 `evaluate_batch.py::evaluate_model()` 按样本计算，验证集参与训练过程的模型选择，因此不是独立测试集。以下是今日记录的实验结果；本次笔记核对没有重新训练模型。

| 指标 | 白噪声 | 低通噪声 |
| --- | ---: | ---: |
| 样本数量 | 10 | 10 |
| 平均输入 SNR | 5.00 dB | 5.00 dB |
| 平均输出 SNR | 8.13 dB | 4.17 dB |
| 平均 SNR 提升 | +3.13 dB | -0.83 dB |
| 最小提升 | +1.02 dB | -1.76 dB |
| 最大提升 | +5.55 dB | +0.18 dB |
| 改善样本数 | 10/10 | 2/10 |

白噪声结果复现了此前记录的 +3.13 dB；两种条件的平均 $\Delta\mathrm{SNR}$ 相差 3.96 dB。**实际观察**是该残差 CNN 在当前低通噪声实验中的 SNR 表现明显差于白噪声。可能解释仍待验证：训练所用白噪声与评估时低通噪声的频谱分布不同；低通噪声也可能与语音本身的低频成分重叠、增加降噪难度。本实验不能证明或区分这两种机制。实验只涉及合成噪声和 10 条验证语音，且这些语音来自用于训练期间模型选择的验证集，不能视为独立测试集，也不能据此推广到真实汽车发动机噪声。

### LibriSpeech 独立测试集与说话人抽样

本节记录固定残差模型在 LibriSpeech `test-clean` 上的两次小样本测试。与 `dev-clean` 的训练/验证划分不同，`test-clean` 不调用 `split_by_speaker()`，用于模型权重固定后的独立评估。用户提供的测试集清单包含 2620 条 FLAC、40 位说话人。两次实验均使用 `ResidualDenoisingCNN`、`output/best_residual_model.pth`、16 kHz、每条 2 秒、输入 SNR 5 dB，并比较 `white` 和 `low_frequency` 合成噪声。以下数值根据本次提供的实验记录整理；实验运行日期和环境未提供，本次文档整理没有重新运行模型。

**实验 A：按排序取前 10 条。** 这 10 条全部来自说话人 `1089`，所以它只能反映单一说话人的小样本表现。

| 噪声类型 | 平均输入 SNR | 平均输出 SNR | 平均 SNR 改善 | 平均输入 MSE | 平均输出 MSE | 改善样本 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 白噪声 | — | — | +5.58 dB | — | — | 10/10 |
| 低频噪声 | 5.00 dB | 5.11 dB | +0.11 dB | 0.001160 | 0.001130 | 7/10（70%） |

白噪声行只记录了本次提供的平均改善和改善数量；未提供的输入/输出均值不补猜。单一说话人抽样不能充分评价跨说话人泛化。

**实验 B：按说话人固定抽样。** 从 40 位说话人中用 `seed=42` 不放回选择 10 位不同说话人，再从每位说话人的文件中抽取一条。抽样只执行一次，两种噪声条件复用同一批语音。

| 噪声类型 | 平均输入 SNR | 平均输出 SNR | 平均 SNR 改善 | 平均输入 MSE | 平均输出 MSE | 改善样本 | 改善比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 白噪声 | 5.00 dB | 9.01 dB | +4.01 dB | 0.001172 | 0.000469 | 8/10 | 80% |
| 低频噪声 | 5.00 dB | 4.45 dB | −0.55 dB | 0.001172 | 0.001289 | 4/10 | 40% |

**观察与限制：** 这次抽样下，固定模型对白噪声的平均 SNR 有明显提升；对低频噪声，平均 SNR 下降 0.55 dB，表现较弱。按文件排序取前 10 条会造成说话人集中；按说话人抽样改善了说话人覆盖，但 10 位说话人、每人一条仍是小样本。相同语音在不同噪声条件下复用，有助于减少语音样本差异对比较的影响。结果只适用于这批测试语音、固定权重和合成噪声，不能据此推断真实汽车发动机噪声或车厢环境的表现。后续可增加说话人数和每位说话人的样本数，再学习双麦克风 TDOA、GCC-PHAT 与波束形成。

### Parseval 定理与频带 MSE

对长度为 $N$ 的实值误差序列 $e[n]$ 及其未归一化 DFT $E[k]$，Parseval 定理给出

$$
\sum_{n=0}^{N-1}|e[n]|^2=\frac{1}{N}\sum_{k=0}^{N-1}|E[k]|^2.
$$

`np.fft.rfft()` 只返回非负频率。令 `power = np.abs(np.fft.rfft(error)) ** 2`，要在频点功率求和时恢复完整双边频谱能量：偶数长度执行 `power[1:-1] *= 2`（DC 和 Nyquist 不加倍），奇数长度执行 `power[1:] *= 2`（只有 DC 不加倍）。这里的 $N$ 是时域误差长度，不是 `rfft` 的频点数。若 $P[k]$ 表示经此修正的功率数组，某频带的时域误差平方和与按样本归一化的频带 MSE 分别为

$$
E_{\mathrm{band}}=\frac{1}{N}\sum_{k\in\mathrm{band}}P[k],
\qquad
MSE_{\mathrm{band}}=\frac{E_{\mathrm{band}}}{N}.
$$

因此代码中的 `band_energy` 实际是该频带误差的 Parseval 能量（平方和），而函数返回的 `band_mse` 才是按时域长度归一化的频带 MSE。`audio_utils.py::calculate_band_and_total_mse(predicted, clean, sample_rate, low_freq, high_freq)` 先对 `predicted-clean` 做 FFT，再返回指定频段 MSE 和 `np.mean(error ** 2)` 得到的整体 MSE。频段使用左闭右开区间 `[low_freq, high_freq)`；当上限等于 Nyquist 频率 `sample_rate / 2` 时，最后一个频段改为闭区间以包含 Nyquist 点。相邻频段因此不会重复计入边界点。

本实验使用 16 kHz 采样率，将 0–8000 Hz 划为 0–500、500–1000、1000–3000、3000–8000 Hz 四段。对长度 16,000 的随机误差核对，四个频段 MSE 之和为 `0.01008350`，整体 MSE 为 `0.01008350`，`np.allclose` 检查为 `True`。这是频带划分与归一化的一致性检查，不是模型效果实验。

### 残差 CNN 的四频段误差分析

分析脚本 `evaluate_frequency_error.py` 使用 `ResidualDenoisingCNN` 和 `output/best_residual_model.pth`，对 LibriSpeech `dev-clean` 的同一 10 条验证语音分别构造白噪声和低频噪声输入。每条语音裁剪或补零至 2 秒（32,000 点），采样率 16 kHz，目标输入 SNR 5 dB。脚本在每条样本上只执行一次模型推理，并复用该输出计算四频段误差、整体误差和噪声估计量。以下是实验记录，不是当前重新运行结果。定义误差变化量为

$$
\Delta MSE=MSE_{\mathrm{output}}-MSE_{\mathrm{input}}.
$$

负值表示误差降低，正值表示误差增加。表中各频段及整体值均为 10 条样本逐条 MSE 的算术平均；整体是时域整体 MSE 的样本平均。

**白噪声：**

| 频段 | 输入 MSE | 输出 MSE | 变化量 |
| --- | ---: | ---: | ---: |
| 0–500 Hz | 0.000152 | 0.000149 | −0.000003 |
| 500–1000 Hz | 0.000154 | 0.000123 | −0.000031 |
| 1000–3000 Hz | 0.000614 | 0.000574 | −0.000039 |
| 3000–8000 Hz | 0.001536 | 0.000398 | −0.001138 |
| 整体 | 0.002455 | 0.001244 | −0.001211 |

白噪声条件下，3000–8000 Hz 的误差降低量约占整体 MSE 降低量的 94%。

**低频噪声：**

| 频段 | 输入 MSE | 输出 MSE | 变化量 |
| --- | ---: | ---: | ---: |
| 0–500 Hz | 0.002058 | 0.001929 | −0.000129 |
| 500–1000 Hz | 0.000198 | 0.000156 | −0.000042 |
| 1000–3000 Hz | 0.000140 | 0.000553 | +0.000413 |
| 3000–8000 Hz | 0.000060 | 0.000348 | +0.000289 |
| 整体 | 0.002455 | 0.002986 | +0.000530 |

低频噪声条件下，模型降低了 0–1000 Hz 的误差，但 1000–8000 Hz 的增加量超过低频改善；其中 1000–3000 Hz 增加最多。两种噪声条件下，输入和输出的四频段 MSE 之和与整体 MSE 的比较均通过 `np.allclose`，记录为 `True`。这些频带变化描述的是重建误差的分布，不单独说明误差增加具体来自噪声残留还是语音失真。

### 真实噪声、预测噪声与频带相关性

`audio_utils.py` 是通用计算和当前复用抽样逻辑的入口，提供参考波形 SNR、SNR 提升、整体 MSE、频带 MSE、频带能量、频带归一化内积及 `sample_files_by_speaker()`。频带 MSE 的 FFT 实现也位于该模块，实验脚本不再各自维护单边补偿、Parseval 归一化和边界规则。`calculate_band_energy()` 返回选定频段的均方能量；`calculate_band_correlation()` 可直接比较两个等形状的一维信号；调用方已有各信号及差值能量时，可用 `calculate_band_correlation_from_energies()` 避免重复 FFT。`evaluate_test.py` 负责测试文件、模型推理、逐条指标及统计输出；模型结构、训练和其他实验配置仍留在各自代码文件。噪声生成与混合保留在 `noise_type.py`、`mix_audio.py`。

```python
from audio_utils import (
    calculate_snr,
    calculate_snr_improvement,
    calculate_mse,
    calculate_band_and_total_mse,
    calculate_band_energy,
    calculate_band_correlation,
    calculate_band_correlation_from_energies,
)

snr_db = calculate_snr(clean, estimate)
band_mse, total_mse = calculate_band_and_total_mse(
    predicted=estimate,
    clean=clean,
    sample_rate=16000,
    low_freq=1000,
    high_freq=3000,
)
```

集中维护这些相同定义的计算，可以避免某个评估脚本使用不同的零值保护、FFT 端点补偿或频带边界，而让实验间的指标失去可比性。频带归一化内积不去均值，不是 Pearson 相关系数；对每条样本先算再平均，也不等于先平均能量再算相关性。

残差模型的输出关系是 $\hat{s}=x-\hat{n}$，其中 $x$ 是混合语音，$\hat{s}$ 是模型输出干净语音估计，$\hat{n}=x-\hat{s}$ 是模型隐含的预测噪声。由 $x=s+n$，真实加入噪声为 $n=x-s$；输出重建误差为

$$
e = \hat{s} - s = n - \hat{n}
$$

将同一频带内信号的平方范数记为能量，有

$$
E_e = E_n + E_{\hat{n}} - 2C
$$

其中 $C$ 是该频带的交叉能量，$\rho$ 是频带归一化内积。若以 $E_a$、$E_b$、$E_{a-b}$ 表示同一频带、同一归一化方式下的能量，则

$$
C = \frac{E_a + E_b - E_{a-b}}{2}
$$

$$
\rho = \frac{C}{\sqrt{E_a E_b}}
$$

这是未去均值的频带内积归一化，不是 Pearson 相关系数；它也不能解释为语音能量被删除的百分比。`evaluate_frequency_error.py` 分别计算两种关系：真实噪声 $n=x-s$ 与预测噪声 $\hat{n}=x-\hat{s}$ 的相关性，差值信号为 $n-\hat{n}=\hat{s}-s$；以及预测噪声 $\hat{n}$ 与干净语音 $s$ 的相关性，差值信号为 $\hat{n}-s$。脚本从已算出的对应频带能量和差值能量取得每条语音的系数，再分别对有限值取算术平均。

1000–3000 Hz 的历史记录如下；噪声能量列是 10 条语音逐条频带能量的平均，能量比是两列均值之比，相关系数则是逐样本相关系数的平均：

| 指标 | 白噪声 | 低频噪声 |
| --- | ---: | ---: |
| 真实噪声平均频带能量 | 0.000614 | 0.000140 |
| 预测噪声平均频带能量 | 0.000946 | 0.000600 |
| 预测/真实能量比 | 1.54 | 4.29 |
| 真实噪声与预测噪声平均相关系数 | 0.6624 | 0.3494 |
| 预测噪声与干净语音平均相关系数 | 0.6647 | 0.8671 |
| 有效样本数 | 10 | 10 |

上述两类相关性都描述 1000–3000 Hz 内信号的归一化内积，不代表模型删除了相同比例的语音能量。结果来自 10 条验证语音及固定权重，只是该实验条件下的观察；验证集参与模型选择，不能据此推断独立测试集、主观音质或真实噪声表现。

白噪声条件下预测噪声与真实噪声的频带相关性较高；低频噪声条件下预测噪声能量相对真实噪声明显偏大，二者相关性也较低。预测噪声与干净语音的相关性在低频噪声条件下更高，说明这两种信号在该频带内有较强的归一化内积，但不能单凭该系数断定模型误删了多少语音。预测/真实能量比也不等于噪声估计误差比；这些观察不足以证明具体失真机制。

### 频带实验的适用范围与后续方向

这些结果来自参与模型选择的验证集，不是独立测试集；`low_frequency` 是由一阶递推生成的合成低通噪声，不代表真实车内发动机噪声。结论只适用于本次固定模型权重、10 条语音、2 秒片段、16 kHz 采样率及 5 dB 输入 SNR 等实验条件。后续可进一步分析语音失真的来源，比较不同噪声类型或混合噪声训练效果，并使用独立测试集和真实车内噪声验证。

### 实验结论与局限

本练习把可控加性噪声、时域监督目标、卷积感受野和参考波形 SNR 串成了可解释流程。历史验证结果和本节新增的独立测试小样本都受样本量、噪声类型和固定权重限制，不能据此声称稳定泛化或主观质量提升。`train.py` 仍只训练残差模型；三模型比较所需另两份权重尚无当前训练入口生成。`analyze_spectrum.py` 与 `evaluate.py` 的 WAV 命名不匹配。该单通道系统也没有阵列几何、通道同步或目标区域标签，因此只为后续区域拾音建立信号处理和学习基础。

## 本地文件与版本控制

`data/` 中的数据集副本和 `output/` 中的权重、音频及图表都由 `.gitignore` 排除。可提交代码和本 README；提交前用 `git status --short` 检查，并确认这些本地文件没有被此前跟踪。其他学习项目的数据和输出规则见仓库根目录的 [`data/README.md`](../../data/README.md)。
