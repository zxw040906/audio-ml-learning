# 音频与机器学习学习记录

这是一个以可运行小练习为主的学习仓库，覆盖信号处理、网络结构、梯度训练和音频数据管线。先在子项目 README 理解概念与参数，再对照脚本注释和输出复习；各目录都标明当前实现与尚未完成的部分。

## 仓库地图

| 路径 | 学习内容 | 从哪里开始 |
| --- | --- | --- |
| `learning/neural-network/` | 感知机、多层前馈网络、卷积、池化和图像张量 | `neuron.py` → `muti_layer.py` → `conv1.py` / `conv2.py` |
| `learning/speech-signal-processing/` | 时域特征、频谱、STFT 和可视化 | `Time_Process/` 与 `Frequency_Process/` |
| `learning/audio-deep-learning/` | Tensor、Dataset/DataLoader、音频变换、CNN 与 TensorBoard | `tensor.py` → `dataset.py` → `dataloader.py` |
| `learning/network_training/` | 损失函数、手动梯度、Autograd 与梯度下降 | `sine_manual_grad.py` → `auto_grad.py` → `manual_grad_nn.py` |
| `projects/area-pickup/` | 区域拾音方向的后续工程 | `docs/area-pickup/overview.md` |
| `learning/common/paths.py` | 数据、素材和输出目录的统一约定 | 见“路径与本地文件” |
| `requirements/` | 按学习项目生成的 Python 依赖清单及统一运行入口 | 见“依赖和运行” |

## 复习路线

1. [语音信号处理](learning/speech-signal-processing/README.md)：从波形、采样率和分帧出发，练习幅值包络、RMS、ZCR、STFT 和谱质心；对照手写实现与 librosa 的边界处理。`SC_BW.py` 尚未计算谱带宽。
2. [神经网络基础](learning/neural-network/README.md)：从感知机、NumPy 多层前向传播，走到激活函数、卷积、池化和图片张量；用 [CNN 参数模板](learning/neural-network/README.md#最简-cnn-模板)按结构图填写参数并检查输出形状。当前主要验证前向计算，尚未训练 CNN。
3. [网络训练](learning/network_training/README.md)：用多项式拟合 `sin(x)`，从手动求导过渡到 Autograd 和 `nn.Linear`，理解平方误差、梯度清零、参数更新，以及学习率过大造成的数值发散。
4. [音频深度学习](learning/audio-deep-learning/README.md)：练习 Tensor、YESNO Dataset/DataLoader、重采样和定长、UrbanSound8K Mel 谱图及 TensorBoard。分类基线尚未完成，需要先确定官方 fold 划分。

四份子项目 README 使用相同的阅读顺序：**学习路线与代码 → 核心概念 → 参数影响与常见问题 → 运行与复习**。具体知识点和参数解释以各项目 README 为准。数据准备见[数据说明](data/README.md)；[区域拾音](projects/area-pickup/README.md)仍是规划项目，需求与实验记录分别放在 `docs/area-pickup/`。

### 后续学习计划

1. 统一信号特征的时间轴和单位，在波形或谱图上叠加特征曲线，记录手写实现与库函数的差异。
2. 将教学脚本中的顶层演示逐步整理为可复用函数，同时保留便于复习的小例子。
3. 基于 UrbanSound8K 的官方 fold 建立分类实验，记录数据划分、随机种子、模型、指标和环境版本。
4. 区域拾音先明确场景、阵列几何、通道同步、采样参数、目标区域和评价指标，再比较算法。
5. 当某个方向形成稳定入口和独立维护需求时，再考虑拆分仓库。

## 环境与依赖

**当前开发与运行环境：Windows 上的 WSL2（Linux）**。仓库和 Conda 环境位于 WSL2 文件系统中；下文的 `python`、`conda`、`tensorboard` 命令都在 WSL2 终端执行。命令示例默认先进入仓库根目录，因此路径可直接复制；在 VS Code 中运行时，选择 WSL2 内对应的 Python 解释器。Windows 原生 Python 环境与这里的依赖和路径不共用。

仓库沿用 WSL2 中已有的 Conda/Python 环境，不提交环境目录。现有依赖清单覆盖神经网络、信号处理和音频深度学习三个目录；新增的网络训练目录尚无独立清单，首次通过统一入口运行其脚本时会生成 `requirements/network_training.txt`。清单由 `requirements/sync_requirements.py` 扫描项目导入并记录当前环境中的版本；它不是跨机器锁定文件，换 Python 版本或环境后应重新生成并检查。

先激活你已经配置好的环境，然后在仓库根目录运行脚本：

```bash
conda activate <你的环境名>
python requirements/sync_requirements.py learning/neural-network/conv1.py
```

把最后的脚本路径换成目标文件即可。该入口会先更新该子项目的依赖清单，再用当前 Python 运行脚本；脚本参数也可跟在路径后面。VS Code 可以使用“同步依赖并运行当前 Python 文件”启动配置。需要安装清单中的包时，使用当前激活环境的 Python：

```bash
python -m pip install -r requirements/neural-network.txt
```

### 两种脚本启动方式

**直接运行**不会同步依赖清单。使用共享路径模块的现有脚本在导入前设置仓库根目录；同目录辅助模块由脚本所在目录解析。仓库根目录执行：

```bash
python learning/audio-deep-learning/dataloader.py
```

如果当前终端不在仓库根目录，先 `cd` 到克隆下来的仓库目录，再执行上面的相对路径命令；`python` 会使用当前激活的环境。

**统一入口运行**会先同步该子项目的依赖清单，再执行脚本：

```bash
python requirements/sync_requirements.py learning/audio-deep-learning/dataloader.py
```

两种方式都会使用当前选定的 Python 环境。第一次使用环境时，先按对应项目清单安装依赖；运行器只同步清单，不会自动安装或升级包。PyTorch、TorchAudio、TorchVision 需要彼此兼容；GPU/CUDA 版本按本机驱动和 PyTorch 安装选择。同步清单读取当前环境实际安装的版本，因此运行前应确认激活了正确的环境。

## 路径与本地文件

`learning/common/paths.py` 提供 `REPO_ROOT`、`DATA_ROOT`、`OUTPUT_ROOT`、`URBANSOUND8K_ROOT`、YESNO 和信号处理素材目录。默认数据/输出位于仓库根目录下的 `.local-data/` 和 `.local-output/`，由 `.gitignore` 排除。可用 `AUDIO_LEARNING_DATA`、`AUDIO_LEARNING_OUTPUT`、`URBANSOUND8K_ROOT` 覆盖默认目录。`output_dir(project, ...)` 会创建对应输出目录。

数据集目录和 UrbanSound8K 下载校验步骤见 [`data/README.md`](data/README.md)。不要把原始录音、大型数据集、模型权重或 TensorBoard 日志提交到 Git。

## 学习建议

可以按“观察输入形状 → 手算一个小例子 → 运行脚本 → 改一项参数 → 解释输出变化”的顺序学习。信号处理的图要结合采样率、帧长和帧移解释；神经网络的输出要结合 batch/channel/spatial 维度解释。当前文件多为教学脚本，直接运行会执行顶层演示代码，有些脚本需要本地数据或会弹出绘图窗口。

具体练习目标、参数影响、运行入口和已知限制见四份子项目 README。

## 记录约定

新增练习时，在对应子项目 README 记录代码入口、输入输出形状、可修改参数及其影响、数据路径、实际结果和遇到的问题。功能尚未实现时明确标注；注释与代码不一致时，先按代码的实际行为记录并说明差异。较长的代码模板和图片放在相关子项目 README，根目录保留路线与入口。

实验记录可复制以下清单填写：

```text
日期与代码版本：
问题或假设：
数据来源与划分：
硬件、采样率、通道数：
帧长、帧移、窗或特征参数：
方法、模型与依赖环境：
运行命令：
结果与评价指标：
观察、限制与下一步：
```
