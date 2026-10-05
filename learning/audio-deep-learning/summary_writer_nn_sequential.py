from torch import nn
import torch

class MyNet(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.module = nn.Sequential(nn.Conv2d(3,32,5,padding=2),nn.MaxPool2d(2),
                                    nn.Conv2d(32,32,5,padding=2),nn.MaxPool2d(2),
                                    nn.Conv2d(32,64,5,padding=2),nn.MaxPool2d(2),
                                    nn.Flatten(),nn.Linear(64*4*4,64),nn.Linear(64,10))
    def forward(self, input):
        output = self.module(input)
        return output

mynn = MyNet()
input = torch.Tensor(1,3,32,32)
output = mynn(input)
print(mynn)
print(output.shape)