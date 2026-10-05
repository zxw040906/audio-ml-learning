# 音频与机器学习学习记录

个人音频信号处理、神经网络与深度学习学习总仓库。各主题目录保持相对独立，便于持续记录，也便于将来将成熟项目拆成单独仓库。

## 学习项目

- `learning/neural-network/`：神经元、卷积、池化等基础练习。
- `learning/speech-signal-processing/`：时域、频域与 STFT 练习。
- `learning/audio-deep-learning/`：PyTorch、音频数据集与 TensorBoard 学习代码。
- `projects/area-pickup/`：后续区域拾音开发；在 `docs/area-pickup/` 记录方案和实验。

## 本地数据与产物

大型数据集、原始音频、模型权重、TensorBoard 日志和 Python 环境不提交到 Git。需要运行时才在仓库根目录创建 `.local-data/`、`.local-models/` 或 `.local-output/`；这些路径已被 `.gitignore` 排除。数据集来源与使用说明见 `data/README.md`。

## 环境与运行

不同学习项目使用各自依赖清单，见 `requirements/`。沿用已有 Conda 环境，仓库不包含环境目录。数据和输出路径通过 `learning/common/paths.py` 统一配置。

## 依赖清单与运行方式

依赖同步逻辑只保留在 `requirements/sync_requirements.py`。在仓库根目录运行项目脚本时使用：

```bash
python requirements/sync_requirements.py learning/neural-network/conv1.py
```

将示例路径换成要运行的 `.py` 文件即可；新建 Python 文件无需添加钩子。也可以在 VS Code 的“运行和调试”中选择 **同步依赖并运行当前 Python 文件**，它会同步当前项目依赖后运行打开的文件。请先激活正确的 Conda/Python 环境。直接执行 `python path/to/file.py` 不会触发同步。
