"""
定义有三个输入的感知元
1.定义感知元函数,线性:y=wx+b 非线性sigmod:y=1/(1+math.exp(-y))
2.调用该函数,确定输入值与每一个输入对应的权重与偏差值
3.得到输出
"""
import math

#1.定义感知元函数,线性:y=wx+b 非线性sigmod:y=1/(1+math.exp(-y))
def neuron(input,weight,bias):
    output = 0
    for x,w in zip(input,weight):
        output += w*x
    output +=bias
    output = 1/(1+math.exp(-output))
    return output
#2.调用该函数,确定输入值与每一个输入对应的权重与偏差值
input = [1,2,3]
weigh = [0.1,0.2,0.3]
bias = 0.02
output = neuron(input,weigh,bias)
print("output=",output)