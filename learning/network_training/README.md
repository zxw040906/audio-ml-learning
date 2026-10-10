# 神经网络与训练：复习笔记

这份笔记把仓库中的基础网络练习和今天的训练代码连成一条学习路线：先看一个神经元如何计算，再看多层结构如何组合，最后理解模型如何用损失和梯度更新参数。代码是教学示例，重点是计算过程，不是完整的生产训练框架。

## 学习路线与代码

| 顺序 | 代码 | 本次要回答的问题 |
| --- | --- | --- |
| 1 | `../neural-network/neuron.py`、`muti_layer.py` | 神经元怎样计算，多层如何传递？ |
| 2 | `../neural-network/conv1.py`、`conv2.py`、`pooling.py` | 卷积与池化如何改变特征形状？ |
| 3 | `sine_manual_grad.py` | 损失函数对各参数的偏导如何写出？ |
| 4 | `auto_grad.py` | Autograd 怎样保存梯度，为什么要清零？ |
| 5 | `manual_grad_nn.py` | `nn.Linear` 的权重和偏置怎样对应多项式参数？ |
| 6 | `loss_func.py` | L1 与 MSE 损失如何计算，`reduction` 如何影响结果？ |
| 7 | `auto_param_nn.py` | 如何用 `nn.Linear` 和 RMSprop 拟合三次多项式？ |
| 8 | `train.py` | 如何按配置搭建 CNN，并完成一次参数更新？ |

## 核心概念：从神经元到网络

### 单个神经元：加权和、偏置、激活函数

在 [`neuron.py`](../neural-network/neuron.py) 中，神经元先计算输入与权重的加权和，再加偏置：

```text
z = Σ(xᵢwᵢ) + b
output = activation(z)
```

权重 `w` 控制每个输入对结果的影响，偏置 `b` 移动整体响应范围。代码使用 Sigmoid：`1 / (1 + exp(-z))`，把数值压到 `(0, 1)`。这一步将线性计算变成非线性输出。

### 多层网络：逐层前向传播

[`muti_layer.py`](../neural-network/muti_layer.py) 用 NumPy 权重矩阵串起多层：每层计算 `data @ weight + bias`，再经过 Sigmoid，并将结果传给下一层。它展示网络如何逐层组合特征，但权重只是随机生成，没有训练，因此输出没有学到的含义。

PyTorch 的 [`nn_module.py`](../neural-network/nn_module.py) 和音频深度学习中的 [`summary_write_nn.py`](../audio-deep-learning/summary_write_nn.py) 则用 `nn.Module` 把层组织成模型。`__init__` 声明层，`forward` 描述数据经过这些层的顺序。`nn.Linear(in_features, out_features)` 对最后一维做可学习的线性变换。

### 为什么需要激活函数

如果多层网络只有线性层，连续线性变换仍可合并成一个线性变换，网络深度本身不会增加表达能力。ReLU、Sigmoid、Tanh 等激活函数为网络加入非线性。

[`nonliner.py`](../neural-network/nonliner.py) 用图片像素演示了激活函数。ReLU 计算 `max(0, x)`，负值变为 0、正值保留；Sigmoid 输出范围为 `(0, 1)`；Tanh 输出范围为 `(-1, 1)`。它们不是按“二分类/多分类”简单固定使用：二分类常在单个输出上用 Sigmoid，多分类常在多个类别分数上用 Softmax；隐藏层常见 ReLU。具体选择取决于任务和损失函数。深层网络中，Sigmoid/Tanh 在饱和区梯度很小，可能让梯度消失；ReLU 对负值梯度为 0，部分神经元可能长期不激活。

## 核心概念：卷积与池化

### 卷积层做什么

[`conv1.py`](../neural-network/conv1.py) 和 [`conv2.py`](../neural-network/conv2.py) 用小张量演示卷积的窗口计算。图像输入通常按 `[batch, channel, height, width]` 排列。卷积核在高、宽方向滑动，对局部区域做加权计算，输出新的特征图。

- `in_channels` 是输入通道数；RGB 图片通常为 3。
- `out_channels` 是卷积核数量，也就是生成的特征图数量。
- `kernel_size` 是每次计算覆盖的局部窗口。
- `stride` 是窗口每次移动的步长；步长增大通常让输出变小。
- `padding` 是输入边缘补的像素，用于控制边界信息和输出尺寸。

卷积核参数在不同空间位置共享，因此可以用较少参数检测边缘、纹理等局部模式。PyTorch 的 `conv1d/conv2d` 数学上执行互相关（不翻转卷积核），学习中通常统称卷积。

`conv2.py` 注释里的输出尺寸公式为：

```text
after_dim = (before_dim - kernel_size + 2 × padding) / stride + 1
```

给定目标输出尺寸时，可移项求 padding：

```text
padding = ((after_dim - 1) × stride - before_dim + kernel_size) / 2
```

对称 padding 要求结果为非负整数。[`nn_module.py`](../neural-network/nn_module.py) 按结构图填写输入、卷积输出通道、卷积核、步长和目标输出高宽，再自动计算 padding 与 Flatten 后的全连接输入宽度。对应的可复制模板在 [神经网络 README](../neural-network/README.md#最简-cnn-模板)。

### 池化层做什么

[`pooling.py`](../neural-network/pooling.py) 演示最大池化与均值池化。池化没有需要训练的权重：最大池化保留窗口中最大的响应，均值池化取局部平均。它通常降低特征图的高宽，减少后续计算，并让表示对小范围位移更稳定；代价是丢失部分空间细节。简单记忆：卷积提取局部特征，池化压缩并汇总局部特征。

例如 32×32 特征图经过一次 `kernel_size=2, stride=2` 池化会变成 16×16。卷积是否改变空间尺寸还要结合卷积核、步长和 padding 计算。

## 核心概念：用多项式逼近 `sin(x)`

训练练习不是直接训练一个复杂 CNN，而是先用简单模型观察参数如何变化。数据来自 `x ∈ [-π, π]` 和目标 `y = sin(x)`。模型用三次多项式逼近它：

```text
y_pred = a + b*x + c*x² + d*x³
```

`a、b、c、d` 是模型参数。这里的 `x、x²、x³` 是输入特征，参数决定每个特征的系数。三次多项式不能在整个区间精确表示正弦函数，所以即使训练正常，损失也不会严格降到 0。

### 训练循环的四步

1. **前向计算**：用当前参数计算 `y_pred`。
2. **计算损失**：将 `y_pred` 和真实值 `y` 比较。
3. **反向求梯度**：求损失对每个参数的导数。
4. **更新参数**：按 `parameter -= learning_rate * gradient` 调整参数。

一个 epoch 是一次完整的训练迭代。本练习每轮都使用全部 2000 个点，因此是全批量梯度下降；循环中的周期性 `print(loss)` 只是观察训练，不参与参数更新。

### 损失函数：误差如何变成一个数

训练文件使用平方误差和：

```text
loss = Σ(y_pred - y)²
```

先平方能避免正负误差互相抵消，较大的错误也会受到更重惩罚。代码中的 `.sum()` 把所有样本误差相加，因此数据点数量越多，损失和梯度通常越大。若改成均方误差 `.mean()`，就是除以样本数的平均损失；这会改变梯度尺度，学习率也应重新观察和调整。

[`loss_func.py`](loss_func.py) 对照了 `L1Loss` 与 `MSELoss`。L1 对误差取绝对值，MSE 对误差平方；较大的误差在 MSE 中受到更重惩罚。`reduction='sum'` 返回所有元素损失之和，`reduction='mean'` 返回平均值，`reduction='none'` 则保留逐元素损失。要注意，脚本注释里的公式若写成除以 `N`，对应的是 `mean`；当前示例显式设为 `sum`，所以没有除以 `N`。

### 手动求梯度：`sine_manual_grad.py`

[`sine_manual_grad.py`](sine_manual_grad.py) 用 NumPy 直接按公式计算梯度。设 `error = y_pred - y`，平方误差和对多项式参数的梯度为：

```text
∂loss/∂a = Σ(2 × error × x⁰)
∂loss/∂b = Σ(2 × error × x¹)
∂loss/∂c = Σ(2 × error × x²)
∂loss/∂d = Σ(2 × error × x³)
```

脚本将这些偏导分别写成 `grad_a` 到 `grad_d`，再手动更新参数。注意 Python 里的 `^` 是按位异或，不是乘方；幂次必须写 `**`。

### Autograd：让 PyTorch 推导梯度

[`auto_grad.py`](auto_grad.py) 将 `a、b、c、d` 创建为 `requires_grad=True` 的 Tensor。PyTorch 在前向计算时记录计算图；调用 `loss.backward()` 后，参数梯度可从各自的 `.grad` 读取。

Autograd 只负责求梯度，不会自动更新参数。代码仍要在 `torch.no_grad()` 中做参数更新，避免把更新操作也记录进下一轮计算图；更新后把 `.grad` 设为 `None`（或调用 `zero_grad()`），否则 PyTorch 会把下一轮梯度累加到旧梯度上。

### 用 `nn.Linear` 管理参数：`manual_grad_nn.py`

[`manual_grad_nn.py`](manual_grad_nn.py) 把 `[x, x², x³]` 作为每个样本的 3 个输入特征，形状为 `[2000, 3]`。`nn.Linear(3, 1)` 为三项特征学习三个权重，并学习一个偏置；这些参数等价于多项式中的 `b、c、d` 和 `a`。Autograd 通过 `loss.backward()` 计算它们的梯度，代码再遍历 `mynn.model.parameters()` 手动更新，并用 `zero_grad()` 清除旧梯度。

[`auto_param_nn.py`](auto_param_nn.py) 展示同一思路下使用优化器的写法。`x.unsqueeze(-1).pow([1,2,3])` 将每个输入变成 `[x, x², x³]` 三列特征；`nn.Linear(3, 1)` 产生一个预测值。脚本使用 `MSELoss()` 和 RMSprop，每轮依次执行前向计算、`zero_grad()`、`backward()` 和 `step()`。使用优化器时，应在每轮反向传播前清梯度，并在反向传播后调用 `step()` 更新参数；优化器应在循环外创建一次，持续跟踪其状态。

## 按配置搭建 CNN：`train.py`

[`train.py`](train.py) 把网络结构图中的信息放进配置区，再由 `MyNetwork` 逐层生成模型。输入图像使用 `[N,C,H,W]` 排列；`CONV_LAYERS` 按顺序记录输出通道数、卷积核、步长和卷积后的目标高宽，`output_size` 指池化之前的尺寸。`POOLS` 用卷积层编号指定在哪些卷积后加入最大池化。

`calc_padding` 根据卷积尺寸公式反推高度和宽度的对称 padding。只有反算结果为非负整数时才能实现目标尺寸，因此部分 stride 与目标尺寸组合会报错。卷积和池化层先放入 `nn.Sequential`，再用一个虚拟输入推导 Flatten 后的特征数，从而自动设置第一层 `Linear` 的输入宽度。`forward` 的数据顺序是卷积/池化、展平、全连接。

示例 CNN 的全连接部分目前是 `Linear → Linear`，两层之间没有激活函数；若希望隐藏层提供非线性，可加入 `nn.ReLU()`。`OUTPUT_UNITS` 是输出维数，并不自动决定任务类型：分类通常将类别分数交给 `CrossEntropyLoss`，目标为每个样本的类别编号；回归可使用 `MSELoss`，目标张量形状需要与预测一致。

训练循环中，外层循环遍历 epoch，内层循环遍历当前 epoch 的各个 batch。每批数据按“前向计算、计算损失、清梯度、反向传播、更新参数”的顺序执行。`optimizer.zero_grad()` 清除前一次迭代留下的梯度；`loss.backward()` 计算当前梯度；`optimizer.step()` 更新参数。模型、损失函数和优化器应在训练循环外创建，避免每个批次重建优化器。当前 `train.py` 用固定张量切片模拟小批量训练；真实项目通常用 `Dataset` 和 `DataLoader` 负责取样、打乱和组批。

### `train.py` 的易错点

- **输入和目标的批次维度必须一致。** 示例有 10 张输入，因此目标也有 10 行；`MSELoss` 下目标形状还要与预测形状 `(N, OUTPUT_UNITS)` 一致。
- **不要在每个批次重新创建优化器。** RMSprop 等优化器可能维护动量等状态；反复重建会丢失这些状态，也会增加不必要开销。把模型、损失函数和优化器放在训练循环外。
- **每轮反向传播前要清梯度。** PyTorch 默认累加 `.grad`；忘记调用 `zero_grad()` 会把不同迭代的梯度相加。典型顺序是先计算 loss，再 `zero_grad()`、`backward()`、`step()`。
- **`step()` 负责更新参数，`zero_grad()` 不负责更新。** 不要把清梯度当成参数更新，也不要只调用 `backward()` 就认为模型已经学习。
- **确认任务类型和标签格式。** 当前示例是回归：目标是浮点数矩阵，使用 `MSELoss`。若改成 10 类分类，模型输出 10 个类别分数，使用 `CrossEntropyLoss`，标签应为形状 `(N,)` 的整数类别编号 `0` 到 `9`，不能把类别 `10` 作为编号。
- **区分 epoch 和 batch。** 一个 epoch 表示模型遍历一次完整训练集；一个 batch 是本次参数更新所使用的一组样本。当前示例外层按 epoch 循环，内层按 `batch_size` 切片，最后不足一个完整批次的样本也会被处理。它用于理解嵌套循环；接入真实数据集时，通常让 `DataLoader` 产生每个 batch，再在内层循环中执行训练步骤。

## 参数影响与常见问题

| 修改项 | 影响 | 需要检查 |
| --- | --- | --- |
| `learning_rate` | 每次参数更新的距离 | 太大可能发散为 `NaN`；太小导致下降缓慢 |
| `learning_num` | 更新次数 | 轮数少可能尚未收敛；轮数多会增加计算，不能修复已发散的训练 |
| 样本数和损失的 `sum/mean` | 损失及梯度尺度 | 从 `sum` 换成 `mean` 后需重新观察学习率 |
| `x` 的范围和多项式次数 | 输入特征与梯度尺度 | 高次幂可能快速放大；改变次数还要同步改变参数和 `Linear.in_features` |
| 参数随机初始化与随机种子 | 每次训练的起点和初始 loss | 不固定种子时打印值会变；比较学习率前先保持初始化一致 |
| `nn.Linear(3,1)` | 输入特征数与每个样本的输出数 | 输入必须是 3 列；输出 `[2000,1]` 与目标 `[2000]` 比较前需处理形状 |
| `reduction='sum'/'mean'` | 损失和梯度的整体尺度 | 样本数或归约方式变化后，重新观察学习率 |
| `optimizer.zero_grad()/step()` | 梯度累加与参数更新 | 每轮反向传播前清梯度，反向传播后更新；优化器在循环外创建 |
| CNN 的 `output_size/POOLS` | 卷积与池化各阶段的空间尺寸 | `output_size` 是池化前尺寸；检查每次池化后的宽高仍为正数 |
| 梯度清零时机 | 下一轮的实际梯度 | 忘记清零会累加旧梯度；在 `backward()` 前错误清零也会改变结果 |

### 学习率与 `NaN`：从这次报错复盘

学习率决定一次参数更新的步长：

```text
new_parameter = old_parameter - learning_rate × gradient
```

- **过大**：更新可能越过低损失区域，造成震荡或发散；参数变得极大后，计算可能溢出为 `inf`，随后出现 `NaN`。
- **过小**：通常更稳定，但损失下降慢，需要更多训练轮数。
- **合适**：损失总体下降且保持有限；不要求每次打印都严格下降，但不应迅速发散。

这次 `manual_grad_nn.py` 的输入包含 `x³`，而 `x` 最大约为 `π`，所以三次特征的绝对值可到约 31，梯度尺度远大于常数项。损失又对 2000 个点求和，进一步放大了梯度。原学习率 `1e-5` 让参数一步更新过远，之后输出出现 `NaN`；改为 `1e-6` 后，loss 能持续下降。这个数值只适用于当前特征和损失定义，不是通用推荐值。

遇到 `NaN` 可以依次检查：

1. 学习率是否让参数一步变化过大。
2. 输入、预测值、损失和梯度是否已经包含 `inf` 或 `NaN`。
3. 损失是 `sum` 还是 `mean`，输入特征的数值范围有多大。
4. 计算中是否有除零、对负数开方或不合法的对数。

输入特征缩放或把损失从 `sum` 改成 `mean` 都会改变梯度大小；做了这些改动后，需要重新检查学习率。不要只因为 loss 数值看起来大就判断有问题，要看它是否有限以及随训练如何变化。

## 运行与依赖

在仓库根目录激活已配置的 PyTorch 环境后运行：

```bash
conda activate <你的环境名>
python learning/network_training/auto_grad.py
python learning/network_training/manual_grad_nn.py
```

`sine_manual_grad.py` 的训练循环可与上述脚本对照，但当前最后一行对 NumPy/Python 标量调用了 `.item()`，在本仓库使用的 NumPy 环境会报错。要运行完整脚本，先把那一行的 `a.item()`、`b.item()` 等换成直接格式化 `a`、`b`、`c`、`d`。`auto_grad.py` 的参数是单元素 PyTorch Tensor，可以调用 `.item()`；它末尾提到 NumPy 的注释沿用了另一份脚本，和实际类型不符。

NumPy 脚本需要 `numpy`；其余两个脚本需要 `torch`。本目录目前没有已提交的独立依赖清单；首次通过 `requirements/sync_requirements.py` 运行项目脚本会生成当前环境对应的 `requirements/network_training.txt`。

## 复习检查

先说出一轮训练的四步，再对照脚本指出每一步所在代码。为什么 `manual_grad_nn.py` 的 `1e-5` 会得到 `NaN`，而 `1e-6` 能下降？`loss.backward()`、`torch.no_grad()` 和 `zero_grad()` 分别负责什么？改变样本数、特征范围或损失归约方式后，学习率为什么要重新观察？
