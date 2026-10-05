# 数据集与素材

## UrbanSound8K（不进入 Git）

完整数据集约 6.0 GB，保存在本机 `.local-data/`，不要提交到仓库。

- 官方 Zenodo 记录及下载：<https://zenodo.org/records/1203745>
- 直接下载：<https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz>
- Zenodo 页面列出的 MD5：`9aa69802bbf37fb986f71ec1483a196e`
- 论文：J. Salamon, C. Jacoby, and J. P. Bello, 《城市声音研究的数据集与分类体系》（原题：A Dataset and Taxonomy for Urban Sound Research），ACM Multimedia，2014 年。使用数据集时请按其记录要求引用。

下载并校验（在仓库根目录运行）：

```bash
mkdir -p .local-data/urbansound8k
curl -L 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz' -o .local-data/urbansound8k/UrbanSound8K.tar.gz
echo '9aa69802bbf37fb986f71ec1483a196e  .local-data/urbansound8k/UrbanSound8K.tar.gz' | md5sum -c -
tar -xzf .local-data/urbansound8k/UrbanSound8K.tar.gz -C .local-data/urbansound8k
```

解压后应设置 `URBANSOUND8K_ROOT` 指向包含 `audio/` 与 `metadata/` 的 `UrbanSound8K` 目录；默认路径为 `.local-data/urbansound8k/UrbanSound8K`。下载前先确认磁盘空间足够。脚本读取该目录的音频和 `metadata/UrbanSound8K.csv`。

## YESNO（小型数据集）

当前工程已有的 60 个 YESNO WAV 已放入 `learning/audio-deep-learning/data/yesno/waves_yesno/`，按你的要求纳入私有学习仓库。这里不包含原始 tar.gz，避免重复保存。

## 信号处理示例音频

现有 `bird.wav`、`time.wav`、`water.wav` 已放入 `learning/speech-signal-processing/assets/audio/`，供练习脚本使用。

图像素材 `nn.png` 和 `zxw.jpg` 已放进各自项目的 `assets/`。

## 路径配置

可用环境变量调整本地数据和输出根目录：`AUDIO_LEARNING_DATA`、`AUDIO_LEARNING_OUTPUT`、`URBANSOUND8K_ROOT`。UrbanSound8K 不会自动下载；小数据集随代码仓库提供。
