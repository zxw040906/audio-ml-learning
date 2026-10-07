#利用torch函数的特性自动求梯度
import torch

# 构造sin函数
x = torch.linspace(start=-torch.pi, end=torch.pi, steps=2000)
y = torch.sin(x)

# 构造预期函数的初始参数
a, b, c, d = torch.rand((),requires_grad=True), torch.rand((),requires_grad=True),\
             torch.rand((),requires_grad=True), torch.rand((),requires_grad=True)

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
    loss = torch.square(input=error).sum()

    #自动反向求梯度
    loss.backward()

    # 未求梯度时更新参数
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        #在进行下一次梯度值更新时,先将此次梯度置为0
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
        
    if epoch % 20 == 0:
        print(f"第 {epoch} 轮,loss={loss:.6f}")

# 打印训练后的参数；NumPy 浮点数可以直接格式化，无需调用 .item()
print(f"y_pred = {a.item()}*x^0 + {b.item()}*x^1 + {c.item()}*x^2 + {d.item()}*x^3")
