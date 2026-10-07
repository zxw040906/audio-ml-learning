# 音频与机器学习学习记录

这是一个以可运行小练习为主的个人学习仓库，记录从离散音频信号分析、神经网络基础到 PyTorch 音频数据管线的学习过程。每个脚本尽量聚焦一个概念；项目 README 解释脚本之间的关系、关键参数与观察重点。

## 仓库地图

| 路径 | 学习内容 | 从哪里开始 |
| --- | --- | --- |
| `learning/neural-network/` | 感知机、多层前馈网络、卷积、池化和图像张量 | `neuron.py` → `muti_layer.py` → `conv1.py` / `conv2.py` |
| `learning/speech-signal-processing/` | 时域特征、频谱、STFT 和可视化 | `Time_Process/` 与 `Frequency_Process/` |
| `learning/audio-deep-learning/` | Tensor、Dataset/DataLoader、音频变换、CNN 与 TensorBoard | `tensor.py` → `dataset.py` → `dataloader.py` |
| `projects/area-pickup/` | 区域拾音方向的后续工程 | `docs/area-pickup/overview.md` |
| `learning/common/paths.py` | 数据、素材和输出目录的统一约定 | 见“路径与本地文件” |
| `requirements/` | 按学习项目生成的 Python 依赖清单及统一运行入口 | 见“依赖和运行” |

## 环境与依赖

仓库沿用本机已有的 Conda/Python 环境，不提交环境目录。三个学习子项目分别使用 `requirements/neural-network.txt`、`requirements/speech-signal-processing.txt` 和 `requirements/audio-deep-learning.txt`。清单由 `requirements/sync_requirements.py` 根据对应目录下的导入扫描当前环境中的已安装发行包版本；它不是跨机器锁定文件，换 Python 版本或环境后应重新生成并检查。

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

**直接运行**不会同步依赖清单，但可以快速执行脚本。脚本会先将仓库根目录加入 Python 导入路径，因此 `from learning.common.paths import ...` 能正常导入共享路径模块；同时脚本目录也会加入导入路径，供 `plot_set.py` 这类同目录模块使用。仓库根目录执行：

```bash
python learning/audio-deep-learning/dataloader.py
```

也可以像下面这样从任意目录使用 Python 和脚本的绝对路径：

```bash
/home/zxw/anaconda3/envs/pytorch/bin/python \
  /home/zxw/python_code/project/audio-ml-learning/learning/audio-deep-learning/dataloader.py
```

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

具体练习目标、变量含义、运行入口和注意事项分别见三个子项目 README。整体后续计划见 [`docs/learning-roadmap.md`](docs/learning-roadmap.md)。
