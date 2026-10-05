import numpy as np

class DeepNet:
    def __init__(self,input_num,out_num,hidden_num):
        self.model = [input_num] + hidden_num + [out_num]
        self.weight = []
        self.layer_num = len(self.model)
        for i in range(self.layer_num - 1):
            current_weight = np.random.rand(self.model[i],self.model[i+1])
            self.weight.append(current_weight)
        self.bias = 0
    def calculate(self,data):
        for w in self.weight:
            y = np.dot(data,w)+self.bias
            y = 1/(1+np.exp(-y))
            data = y
        return y

in_num = 3
out_num = 1
hidden_num = [3,2,3]
mynn = DeepNet(input_num=in_num,out_num=out_num,hidden_num=hidden_num)
input = np.array([1,2,3])
output = mynn.calculate(input)
print("output=",output)