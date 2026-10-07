# 学习路线与记录

这个仓库按“信号表示 → 网络结构 → 梯度训练 → 音频数据管线”组织复习。每次新增练习，记录代码入口、输入输出形状、参数影响、观察到的问题与下一步；四份学习项目 README 是具体的复习入口。

## 已有内容

### 1. 离散语音信号特征

[`learning/speech-signal-processing/`](../learning/speech-signal-processing/README.md) 包含幅值包络、RMS、ZCR、手工和 librosa STFT、谱质心。复习采样率、帧长、帧移、窗函数、频点到 Hz 的换算，以及手写实现与库函数的边界定义差异。`SC_BW.py` 尚未计算谱带宽。

### 2. 神经网络计算基础

[`learning/neural-network/`](../learning/neural-network/README.md) 从 Sigmoid 感知机和 NumPy 多层前向传播开始，再验证线性层、激活函数、一维/二维卷积、池化和图片张量处理。`nn_module.py` 按结构图计算 padding 与 Flatten 宽度；当前示例主要验证前向计算和形状，没有训练 CNN。

### 3. 损失与梯度训练入门

[`learning/network_training/`](../learning/network_training/README.md) 用三次多项式拟合 `sin(x)`，从 NumPy 手动求导过渡到 PyTorch Autograd，再用 `nn.Linear` 管理可训练参数。复习平方误差和、梯度清零、参数更新及学习率过大时的数值发散。

### 4. PyTorch 音频管线

[`learning/audio-deep-learning/`](../learning/audio-deep-learning/README.md) 演示 Tensor、YESNO Dataset/DataLoader、音频长度/采样率规范化、UrbanSound8K Mel 谱图和 TensorBoard 记录。下一阶段在明确官方 fold 拆分后，可将特征、网络和训练循环连接成可评估的分类基线。

## 后续路线

1. 为信号特征统一时间轴和单位，在谱图/波形上叠加特征曲线，并记录手写与库实现差异。
2. 把教学脚本中顶层演示逐步提炼成可复用函数，同时保持小例子方便阅读。
3. 基于 UrbanSound8K 做规范的 train/validation/test 实验，记录 fold、随机种子、模型、指标和环境版本。
4. 区域拾音先完善场景、阵列几何、同步与通道数、采样参数、目标区域和评价指标，再开始算法比较。
5. 某方向形成稳定入口、独立依赖和维护节奏后，再考虑拆为独立仓库。

## 文档记录约定

四个学习目录的 README 遵循同一阅读顺序：**学习路线与代码 → 按主题展开概念 → 参数影响与常见问题 → 运行与依赖 → 复习检查**。新增脚本时，在对应目录补上它的输入与输出形状、可修改参数、改变参数后的影响、数据路径及实际运行结果。代码已有但尚未实现的功能，要直接写明“当前未实现”；发现注释和代码不一致时，以代码当前行为为准并注明差异。图片和较长模板放在最相关的项目 README，根 README 只保留入口与链接。

## 实验记录模板

- 日期与代码版本：
- 问题/假设：
- 数据来源及划分：
- 硬件、采样率、通道数：
- 帧长、帧移、窗或特征参数：
- 方法/模型及依赖环境：
- 运行命令：
- 结果与评价指标：
- 观察、限制与下一步：
