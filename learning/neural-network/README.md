# 神经网络基础练习

本目录用很小的输入验证神经元、全连接层和图像卷积/池化的基本计算。这里的脚本以演示为主，通常在模块顶层直接构造数据并打印或显示结果。

## 建议阅读顺序

1. `neuron.py`：手写一个三输入感知机。先做加权和 `Σ(xᵢwᵢ)+b`，再过 Sigmoid，把线性组合压到 0 到 1 之间。
2. `muti_layer.py`：用 NumPy 权重矩阵串联多个全连接层。每层计算 `data @ weight + bias` 后做 Sigmoid，输入向量依次传播到输出层。
3. `conv1.py`、`conv2.py`：把卷积理解为滑动窗口加权求和，比较步长和 padding 对输出长度/宽高的影响。
4. `pooling.py`：对图像做最大池化或均值池化，观察空间分辨率下降和局部信息汇总。
5. `pic_process_(conv2test).py`：对真实图片应用 `nn.Conv2d`，并把模型图写入 TensorBoard。

## 文件说明与观察点

### 感知机与多层前馈网络

`neuron.py` 的输入、权重都是长度为 3 的列表，`zip` 逐项相乘求和。脚本使用 `math.exp` 实现 Sigmoid。可以手算加权和，再核对打印值。若权重和偏置较大，指数函数可能溢出；生产实现通常使用数值稳定的框架算子。

`muti_layer.py` 中 `model = [input_num] + hidden_num + [out_num]` 表示每层神经元个数，例如 `[3, 3, 2, 3, 1]`。权重矩阵形状为 `(本层宽度, 下一层宽度)`，因此数据按行向量右乘矩阵。此示例随机初始化权重、不训练参数，偏置为所有层共用的标量 0；它展示前向传播，不是可学习的完整训练网络。可尝试固定随机种子、改批量输入形状，理解矩阵维度。

### 一维与二维卷积

`conv1.py` 输入形状为 `[batch, channel, length] = [1, 1, 5]`，卷积核为 `[out_channel, in_channel, kernel_length] = [1, 1, 3]`。`valid` 不补边，只有完全落在输入内的窗口参与运算；`padding=2` 在左右各补两个零。PyTorch `conv1d/conv2d` 实际执行互相关（核不翻转），学习时常统称卷积。

`conv2.py` 的输入 `[1,1,5,5]` 表示 batch、输入通道、高、宽，核 `[1,1,3,3]`。输出每一维按下式计算：

```text
out = floor((in + 2 × padding - kernel) / stride) + 1
```

脚本把 stride 设为 1 或 2、padding 设为 valid 或 1，并打印输出形状。注意代码注释中的通用公式省略了向下取整；维度不是整除时要取 floor。

### 池化与图片卷积

`pooling.py` 读取 `assets/zxw.jpg`，转成 `[C,H,W]` 浮点 Tensor，再增加 batch 维变成 `[1,C,H,W]`。模型当前 forward 使用均值池化；最大池化层也定义了，可替换后比较局部最强响应与局部平均。TensorBoard 图像输出在 `.local-output/neural-network/log_network/`。

`pic_process_(conv2test).py` 对同一图片运行可学习的 `nn.Conv2d(3,3,kernel_size=2,stride=1,padding=1)`，然后把模型图写入 `.local-output/neural-network/log_conv/` 并显示变换后的图片。输出通道数仍为 3，所以可以转换回 RGB 图像；随机初始化的卷积核不会产生有意义的增强效果，它只是展示形状和数据流。

## 运行与依赖

从仓库根目录激活已配置的环境后运行，例如：

```bash
conda activate <你的环境名>
python requirements/sync_requirements.py learning/neural-network/conv2.py
python requirements/sync_requirements.py learning/neural-network/pooling.py
```

依赖清单是 `requirements/neural-network.txt`。首次在新环境使用时可运行 `python -m pip install -r requirements/neural-network.txt`；清单由统一入口根据当前 Python 环境扫描生成，先确认激活的是正确环境。PyTorch、TorchVision 和 TensorBoard 用于张量、图片变换和模型图记录；NumPy 用于多层网络演示；Pillow 用于图片读写。该项目不需要音频数据。`pooling.py` 和图片卷积脚本需要 `assets/zxw.jpg`，运行后会弹出图像窗口。

统一入口负责把仓库根目录加入导入路径，因此脚本能读取 `learning.common.paths` 中的输出目录；不要在每个脚本重复写路径注入或包安装逻辑。
