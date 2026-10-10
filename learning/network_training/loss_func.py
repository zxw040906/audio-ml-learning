"""
损失函数使用:
1.定义两个变量
2.损失函数:平方均值,最大似然,交叉熵
"""
import torch
from torch import nn

y_pred = torch.tensor([1,2,3],dtype=torch.float32)
y = torch.tensor([1,2,5],dtype=torch.float32)

#选择L1loss,参量选择 取值与取和:{(P1-E1)+(P2-E2)+...+(PN-EN)}/N
loss_l1 = torch.nn.L1Loss(reduction='sum')
result1 = loss_l1(y_pred,y)
print(result1)
#选择MSEloss---{(P1-E1)^2+(P2-E2)^2+...+(PN-EN)^2}/N
loss_l2 = torch.nn.MSELoss(reduction='sum')
result2 = loss_l2(y_pred,y)
print(result2)
