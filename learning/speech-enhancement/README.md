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
| `evaluate_noise_type.py` | 用同一残差模型和固定验证文件比较白噪声、低通噪声 |
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

$$y[t]=s[t]+n[t],$$

其中 $s[t]$ 是干净语音，$n[t]$ 是噪声，$y[t]$ 是混合波形；下标 $t$ 表示离散样本索引，样本幅值在本实验中为无量纲浮点数。模型假设噪声与语音按样本相加，未描述混响、非线性削波、时变麦克风响应或阵列空间效应。白噪声在生成模型中是各样本独立同分布的随机噪声；有限长度样本的经验均值和方差不保证恰好为理论值。

对有 $T$ 个样本的序列 $x[t]$，平均平方功率定义为

$$P_x=\frac{1}{T}\sum_{t=0}^{T-1}x[t]^2.$$

它是同一幅值标度下比较能量强弱的功率代理；若是电压且负载相同，物理平均功率与均方电压成正比。由于 `add_noise()` 对等长序列分别取 `np.mean(x**2)`，变量 `clean_power` 和 `noise_power` 就对应这个定义。

### 低通随机噪声与频谱分析

`noise_type.py::generate_low_frequency_noise(length, a=0.95, seed=42)` 使用 `np.random.default_rng(seed)` 创建局部随机数生成器，再由 `standard_normal(length)` 生成长度为 `length` 的标准正态白噪声序列 $w[n]$。逐样本递推

$$v[n]=a v[n-1]+(1-a)w[n],$$

其中实现的初始值为 $v[0]=(1-a)w[0]$。$a=0.95$ 使前一采样状态保留较多、每一步新噪声输入权重较小，因此生成的合成一阶低通随机噪声更平滑、能量更集中于低频；同一长度和种子会生成可复现序列。它是合成噪声模型，并不等同于真实汽车发动机噪声。

频谱分析步骤用 `np.fft.rfft()` 计算实信号单边频谱，`np.fft.rfftfreq()` 生成频率坐标，`np.abs()` 取幅度后平方得到各频点功率，再以 `np.sum()` 汇总 0～500 Hz 和全频带功率。偶数长度实信号的 `rfft` 结果包含 DC 和 Nyquist 端点，它们不加倍；其余正频率功率需乘 2，以补上共轭对称的负频率部分（示例代码为 `one_sided_power[1:-1] *= 2`）。目前 `noise_type.py` 中这段频谱分析示例处于注释状态；下面的比例是当时实验记录，不是运行该文件当前主程序的结果。对 16 kHz 采样率、长度 16,000、种子 42 的噪声，0～500 Hz 能量占比分别为低通噪声 84.30%、白噪声 6.23%。

### 从目标 SNR 推导噪声缩放

目标信噪比 $S$ dB 定义为干净信号功率与缩放后噪声功率的比值：

$$S=10\log_{10}\left(\frac{P_s}{P_{\alpha n}}\right).$$

先将分贝式两侧除以 10 并取 10 次幂：

$$10^{S/10}=\frac{P_s}{P_{\alpha n}}.$$

若缩放系数 $\alpha$ 是实数，缩放后功率为

$$P_{\alpha n}=\frac1T\sum_t(\alpha n[t])^2=\alpha^2\frac1T\sum_tn[t]^2=\alpha^2P_n.$$

代回目标比值并解 $\alpha$：

$$10^{S/10}=\frac{P_s}{\alpha^2P_n}
\quad\Longrightarrow\quad
\alpha^2=\frac{P_s}{P_n10^{S/10}}
\quad\Longrightarrow\quad
\alpha=\sqrt{\frac{P_s}{P_n10^{S/10}}}.$$

取正根是因为只调整噪声幅度，不翻转其符号。最终混合波形为 $y=s+\alpha n$。`mix_audio.py::add_noise` 逐项实现上述运算；输入必须是一维、等长且功率为正。手算检查：若 $P_s=1$、$P_n=4$、目标 $S=0$ dB，则 $\alpha=\sqrt{1/4}=0.5$，缩放噪声功率变为 1，信噪比为 0 dB。目标 SNR 越低，分母中的 $10^{S/10}$ 越小，噪声缩放越大。

独立练习函数 `scale_noise_to_snr(clean, noise, snr_db)` 分别用 `np.mean(clean**2)` 和 `np.mean(noise**2)` 求干净语音与原始噪声的均方功率，按目标分贝值求目标噪声功率，再开平方得到幅度缩放系数；其推导与前文“从目标 SNR 推导噪声缩放”相同。目标 5 dB 的手动核验结果为实际 5.00 dB。该函数用于独立练习和核验；正式 Dataset 路径不先调用它再缩放一次，而是对白噪声或低通噪声只调用 `mix_audio.py::add_noise()` 一次，由该函数负责目标 SNR 缩放和混合。

### MSE 目标与 SNR 指标

训练代码使用波形均方误差。对 $T$ 个样本，模型输出 $\hat s[t]$，干净目标为 $s[t]$，则

$$L_{MSE}=\frac1T\sum_{t=0}^{T-1}(\hat s[t]-s[t])^2.$$

前提是预测与目标对齐且同长度。它惩罚每个采样点的平方误差；`train.py` 使用 `nn.MSELoss()` 默认 mean reduction，对 batch 和时间元素求均值。

评价脚本将误差 $e[t]=\hat s[t]-s[t]$ 视为残留噪声，并计算

$$\mathrm{SNR}_{out}=10\log_{10}\left(\frac{\sum_t s[t]^2}{\sum_t e[t]^2}\right).$$

令 $P_s=T^{-1}\sum_ts[t]^2$、$P_e=T^{-1}\sum_te[t]^2$，则分子分母同除以 $T$，比值不变，故也等于 $10\log_{10}(P_s/P_e)$。`evaluate.py::calculate_snr` 对一条波形计算该值；`evaluate_batch.py::calculate_snr` 沿每条样本的时间轴分别计算。两者均用 `1e-12` 防止误差功率为零时除零。

MSE 与输出 SNR 的分子分母角色不同：对固定目标能量的同一条样本，降低误差 MSE（或误差平方和）会提高输出 SNR；但跨样本时目标能量不同，单看 MSE 不足以排序 SNR。训练还使用批次/样本均值，而脚本计算每条记录的能量比。SNR 提升定义为输出 SNR 减输入混合 SNR，单位 dB；它是参考波形误差指标，不等价于主观听感、语音可懂度或空间拾音效果。

### Conv1d：计算、形状、参数与感受野

模型调用前把 `[B,T]` 变为 `[B,1,T]`，其中 $B$ 是批大小、1 是单声道通道、$T$ 是时间采样点；逐层卷积后再 squeeze 回 `[B,T]`。`model.py` 的各卷积均为 `kernel_size=9`、stride 默认 1、bias 默认启用。dilation 为 $d$ 时，有效核宽为 $k_{eff}=d(k-1)+1$，一维输出长度为

$$T_{out}=\left\lfloor\frac{T+2p-d(k-1)-1}{s}+1\right\rfloor,$$

其中 $p$ 是左右各自的 padding，$s$ 是 stride。代码对每层使用 $p=d(k-1)/2$，本模型的核宽为奇数且 stride=1，因此长度保持为 $T$。

单个输出位置、输出通道 $o$ 的计算是对所有输入通道和核位置加权求和再加偏置：

$$z_{o,t}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{q=0}^{k-1}w_{o,c,q}\,x_{c,ts-p+qd}.$$

该索引形式描述 stride 与 dilation 的采样位置；越界位置由 padding 零补齐。对 `SimpleDenoisingCNN`，第一层第一个窗口是输入 9 点与 9 个卷积权重逐项乘积求和，加一个 bias；ReLU 再逐点执行 $\max(0,z)$。

普通 Conv1d 的参数量是二维卷积公式在核高为 1 时的特例：

$$
N_{\mathrm{params}}=C_{\mathrm{out}}(C_{\mathrm{in}}k+b),
$$

其中 $C_{\mathrm{in}},C_{\mathrm{out}}$ 是输入和输出通道数，$k$ 是核宽（采样点），$b=1$ 对应 `bias=True`、$b=0$ 对应 `bias=False`；此处为 `groups=1`。核参数在时间位置间共享，因此参数数不乘 $T$。原始/残差网络三层参数分别是 $16(1·9+1)=160$、$16(16·9+1)=2320$、$1(16·9+1)=145$，总计 2625。残差模型只是把最后层初值设为零，不改变参数总数。空洞网络五层参数分别为 160、2320、2320、2320、145，总计 7265；dilation 和 padding 不改变权重张量尺寸。

感受野按层计算。设初值 $r_0=1$ 个输入样本、jump $j_0=1$，第 $l$ 层 stride 为 $s_l$、核宽 $k_l$、dilation 为 $d_l$，则

$$j_l=j_{l-1}s_l,\qquad r_l=r_{l-1}+(k_l-1)d_lj_{l-1}.$$

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

$$\hat s=y-\hat n.$$

若初始化末层 weight 与 bias 都为零，则任意支路特征经末层线性映射后 $\hat n=0$，于是初始化时 $\hat s=y$，即模型从恒等映射开始。训练仍通过 `MSELoss(\hat s,s)` 的误差更新网络；对减法关系，$\partial\hat s/\partial\hat n=-1$，故梯度会沿预测噪声支路反传。零初始化只保证最终层输出初始为零，不表示中间层权重为零，也不保证训练后必然去噪成功。实现位于 `ResidualDenoisingCNN.__init__` 与 `forward`。

### 划分、训练、验证与模型选择

`train.py::split_by_speaker` 从 `speaker/chapter/*.flac` 路径取说话人 ID，固定 NumPy 种子 42 打乱说话人，约 20% 说话人放入验证集且至少 1 个；余者用于训练。按说话人隔离可减少同一说话人音色同时进入训练与验证造成的泄漏。随后两个 Dataset 各自按排序后的路径最多取 40/10 条；这不是完整数据集训练。白噪声则由样本 index 决定种子。

每个 epoch 训练集 loader 打乱，验证集不打乱。训练步骤是 forward → batch mean MSE → `zero_grad()` → `backward()` → Adam `step()`；验证用 `model.eval()` 和 `no_grad()`，不更新权重。脚本在 5 个 epoch 中保存验证 MSE 最低时的 `state_dict` 到 `output/best_residual_model.pth`。验证集因被用于挑选 epoch/权重，不能再当独立测试集；最终泛化结论需要另留测试说话人/数据。当前没有独立测试集，也没有在训练入口比较三种架构。

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

### 实验结论与局限

本练习把可控加性噪声、时域监督目标、卷积感受野和参考波形 SNR 串成了可解释流程。历史记录中残差 CNN 的 10 条验证样本均有正提升，但验证集参与模型选择，样本数少，不能据此声称稳定泛化或主观质量提升。当前 `train.py` 只训练残差模型；三模型比较所需另两份权重尚无当前训练入口生成。`analyze_spectrum.py` 与 `evaluate.py` 的 WAV 命名不匹配。该单通道系统也没有阵列几何、通道同步或目标区域标签，因此只为后续区域拾音建立信号处理和学习基础。

## 后续学习计划

### 语音低频能量与噪声频谱重叠

**计划记录：2026-10-11，W03-03。** 分析相同 10 条验证语音在 0～500 Hz 的能量占比，从实现 `calculate_low_frequency_ratio(signal, sample_rate=16000, cutoff=500)` 开始，以进一步检查频谱重叠假设。该函数和实验属于下次计划，尚未实现、尚未验证。之后可考虑改变低通滤波系数 `a` 或研究不同噪声训练策略。

## 本地文件与版本控制

`data/` 中的数据集副本和 `output/` 中的权重、音频及图表都由 `.gitignore` 排除。可提交代码和本 README；提交前用 `git status --short` 检查，并确认这些本地文件没有被此前跟踪。其他学习项目的数据和输出规则见仓库根目录的 [`data/README.md`](../../data/README.md)。
