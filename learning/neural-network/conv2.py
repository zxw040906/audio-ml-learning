"""
维度计算:
after_dim = (befor_dim - kernel_size + 2 * padding) / stride +1
"""
import torch
import torch.nn.functional

#二维卷积
#input -> (Batch_size,Channels,height,weight)->(1,1,5,5)
input = torch.tensor([[[[1,2,3,4,5],
                        [6,7,8,9,10],
                        [11,12,13,14,15],
                        [16,17,18,19,20],
                        [21,22,23,24,25]]]])

kernel = torch.tensor([[[[1,1,1],
                         [2,2,2],
                         [3,3,3]]]])

output1 = torch.nn.functional.conv2d(input,kernel,stride=1,padding="valid")
print("步长为1,不补零:",output1.shape)

output2 = torch.nn.functional.conv2d(input,kernel,stride=2,padding="valid")
print("步长为2,不补零:",output2.shape)

output3 = torch.nn.functional.conv2d(input,kernel,stride=1,padding=1)
print("步长为1,补一个零:",output3.shape)

output4 = torch.nn.functional.conv2d(input,kernel,stride=2,padding=1)
print("步长为2,补一个零:",output4.shape)