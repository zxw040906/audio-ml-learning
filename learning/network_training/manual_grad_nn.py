import torch
from torch import nn

#定义三次项sin函数
x = torch.linspace(start=-torch.pi,end=torch.pi,steps=2000)
y = torch.sin(x)
p = torch.tensor([1,2,3])
input = x.unsqueeze(-1).pow(p)

#定义网络模型
class Liner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=3,out_features=1),nn.Flatten(0,1))
    def forward(self,data):
        output = self.model(data)
        return output

#学习次数
learning_num = 2000
#学习率
learning_rate = 1e-6
mynn = Liner()
for epoch in range(learning_num):
    y_pred = mynn(input)
    error = y_pred - y
    loss = torch.square(input=error).sum()
    
    #求梯度
    loss.backward()
    layer_liner = mynn.model[0]
    layer_flatten = mynn.model[1]
    #更新参数
    with torch.no_grad():
        for parameter in mynn.model.parameters():
            parameter -= learning_rate * parameter.grad
        mynn.model.zero_grad()
    if epoch % 200 == 0:
        print(loss)

print(f"y_pred = {mynn.model[0].bias.item()} + {mynn.model[0].weight[:,0].item()}*x + {mynn.model[0].weight[:,1].item()}*x^2 + {mynn.model[0].weight[:,2].item()}*x^3")