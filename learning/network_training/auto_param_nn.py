from torch import nn
import torch

class Liner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(3,1),nn.Flatten(0,1))
    def forward(self,data):
        output = self.model(data)
        return output

x = torch.linspace(-torch.pi,torch.pi,2000)
y = torch.sin(x)
mynn = Liner()
#x^1,x^2,x^3
p = torch.tensor([1,2,3])
input = x.unsqueeze(-1).pow(p)
#学习率
learning_rate = 1e-4
#学习次数
learning_num = 2000
optimiser = torch.optim.RMSprop(params=mynn.model.parameters(),lr = learning_rate)
#均方根误差
loss_func = torch.nn.MSELoss()
for echo in range(learning_num):
    y_pre = mynn(input)
    loss = loss_func(y_pre,y)
    #梯度置零
    optimiser.zero_grad()
    #求梯度
    loss.backward()
    #优化器
    optimiser.step()
    if (echo %200==0):
        print(f"第{echo}轮学习,loss={loss}")