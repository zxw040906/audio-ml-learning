# 数据集与素材

本仓库把小型教学素材和体量较大的数据集分开管理。大型数据、原始录音、模型权重及运行输出放在 Git 忽略的本地目录中；数据集的来源、目录结构和准备方式记录在这里。

## UrbanSound8K（本地准备，不提交 Git）

UrbanSound8K 含 8732 条城市环境声音剪辑，按 10 个官方 fold 存储，CSV 元数据包含文件名、fold、类别名和 `classID` 等字段。项目脚本读取 `audio/fold1` 至 `fold10` 以及 `metadata/UrbanSound8K.csv`。归档文件约 6 GB，解压还需要额外空间。

- [Zenodo 官方记录与下载](https://zenodo.org/records/1203745)
- [直接下载归档](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz)
- 官方页面所列 MD5：`9aa69802bbf37fb986f71ec1483a196e`
- 论文：J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” ACM Multimedia, 2014。用于研究或公开结果时按数据集记录要求引用。

在仓库根目录下载、校验并解压：

```bash
mkdir -p .local-data/urbansound8k
curl -L 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz' -o .local-data/urbansound8k/UrbanSound8K.tar.gz
echo '9aa69802bbf37fb986f71ec1483a196e  .local-data/urbansound8k/UrbanSound8K.tar.gz' | md5sum -c -
tar -xzf .local-data/urbansound8k/UrbanSound8K.tar.gz -C .local-data/urbansound8k
```

确认解压结果为 `.local-data/urbansound8k/UrbanSound8K/audio/` 和 `metadata/`。默认 `URBANSOUND8K_ROOT` 已指向该目录；如果数据放在其他地方，可设置环境变量，例如 `export URBANSOUND8K_ROOT=/path/to/UrbanSound8K`。`learning/audio-deep-learning/dataset_ubs8k.py` 不会自动下载数据。

UrbanSound8K 官方评估基于 fold 划分。做实验时应保留官方 fold 结构，避免随机将同一录音来源的片段混入训练与验证集合。记录实验时标注使用了哪些 fold、特征参数和随机种子。

## YESNO（仓库内的小型示例）

当前仓库保留 60 个 YESNO WAV，位于 `learning/audio-deep-learning/data/yesno/waves_yesno/`，不需要首次运行时联网下载。示例 Dataset 用它演示逐条读取、波形绘图、重采样和 DataLoader 批处理。上游数据集可由 TorchAudio YESNO 页面了解；这里的学习副本不含原始压缩包。

## 信号处理音频与图像

- `learning/speech-signal-processing/assets/audio/`：`bird.wav`、`time.wav`、`water.wav`，供时域和频域特征练习使用。
- `learning/audio-deep-learning/assets/nn.png`：TensorBoard 图像变换练习。
- `learning/neural-network/assets/zxw.jpg`：卷积与池化图像练习。

这些素材随当前仓库提供。运行前可检查文件存在；Matplotlib/图片显示需要可用的图形后端。

## 统一路径配置

`learning/common/paths.py` 集中管理路径：

| 名称 | 默认位置/用途 | 可配置变量 |
| --- | --- | --- |
| `DATA_ROOT` | 仓库根目录 `.local-data/` | `AUDIO_LEARNING_DATA` |
| `OUTPUT_ROOT` | 仓库根目录 `.local-output/` | `AUDIO_LEARNING_OUTPUT` |
| `URBANSOUND8K_ROOT` | `.local-data/urbansound8k/UrbanSound8K` | `URBANSOUND8K_ROOT` |
| `YESNO_AUDIO_DIR` | 仓库内 YESNO WAV 目录 | 固定仓库相对路径 |
| `SPEECH_AUDIO_DIR` | 信号处理素材音频目录 | 固定仓库相对路径 |

大型数据和输出已在 `.gitignore` 中排除。提交前确认没有意外加入录音、缓存、TensorBoard event 或模型检查点。
