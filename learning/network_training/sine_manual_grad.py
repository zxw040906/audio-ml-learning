import numpy as np

# 构造sin函数
x = np.linspace(start=-np.pi, stop=np.pi, num=2000)
y = np.sin(x)

# 构造预期函数的初始参数
a, b, c, d = np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()

# 定义学习率(决定学习速度,越大越快,越小越慢),此处设置为10^-6
learning_rate = 1e-6
# 定义学习次数
learning_num = 200

# 此处需要多次训练,进行求偏导然后更新参数环节
for epoch in range(learning_num):
    # 每轮先根据当前参数重新计算预测值和损失

    # 构造损失函数:所有样本的平方误差和
    y_pred = a + b * x + c * x**2 + d * x**3
    error = y_pred - y
    loss = np.square(error).sum()

    # 求偏导；^ 在 Python 中是异或，乘方需要使用 **
    grad_y_pred = 2 * error
    grad_a = (grad_y_pred * x**0).sum()
    grad_b = (grad_y_pred * x**1).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # 更新参数
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    if epoch % 20 == 0:
        print(f"第 {epoch} 轮,loss={loss:.6f}")

# 打印训练后的参数；NumPy 浮点数可以直接格式化，无需调用 .item()
print(f"y_pred = {a.item()}*x^0 + {b.item()}*x^1 + {c.item()}*x^2 + {d.item()}*x^3")
