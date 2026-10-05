import torch.nn.functional
import torch

#一维卷积
#input -> (Batch_size,Channels,Length)->(1,1,5)
input = torch.tensor([[[-1,2,1,1,0]]])
#kernel -> (Channel_k,Channels/group,Length_k)->(1,1/1=1,3)
kernel = torch.tensor([[[1,0,-1]]])

#两边不补零 -1,2,1,1,0
output = torch.nn.functional.conv1d(input=input,weight=kernel,stride=1,padding="valid")
print("不补零卷积output=",output)

#两边补两个零 0,0,-1,2,1,1,0,0,0
output1 = torch.nn.functional.conv1d(input=input,weight=kernel,stride=1,padding=2)
print("补零卷积output1=",output1)
