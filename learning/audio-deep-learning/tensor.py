import torch
import numpy as np

#定义tensor的数据类型,使用不同类型定义tensor 
tensor_int = torch.IntTensor([1,2,3])
tensor_float = torch.DoubleTensor([1,2,3])
#只定义形状,无幅值
tensor_size1 = torch.Tensor((3,3,3))#只定义一维,包含3,3,3
tensor_size2 = torch.Tensor(3,3,3)#定义三维3*3*3,一共27个数
#定义完全为0的tensor,形状与input相似
tensor_zero = torch.zeros_like(tensor_size2)
#定义元素为一组在[0,1]内分布的随机数
tensor_random = torch.rand((2,2))
#定义元素为一组满足正态分布(均值为0,方差为1)的随机数
tensor_zt_random = torch.randn((2,2))
#定义返回形状为1*n,元素为0~n内不重复的整形Tensor
tensor_random_int = torch.randperm(5)

#改变Tensor形状调整
tensor_3d1 = tensor_size2.view(1,3,9)
tensor_3d2 = tensor_size2.resize_(1,9,3)
tensor_3d3 = torch.reshape(tensor_size2,(9,1,3))
#Tensor的数据类型,任意位置幅值
#改变数据类型
tensor_float_int = tensor_float.int()
#获取数值
a = tensor_float_int[2]#第一行的第二个值
"""
Tensor 运算
注意此处加减乘除均为对应点运算而非矩阵运算
"""
b = torch.tensor((1,2,3))
c = torch.tensor((4,5,6))
add = torch.add(b,c)
sub = torch.sub(c,b)
mul = torch.mul(b,c)
div = torch.div(c,b)
print("add=",add)
print("sub=",sub)
print("mul=",mul)
print("div=",div)

#幂次
b_3 = b.pow(3)
print("b^3=",b_3)

"""此处为矩阵乘法,各元素单独相乘"""
matmul = torch.matmul(b,c)
print("matmul=",matmul)

#Tensor的拼接,横向,纵向,dim=0表示1维拼接
b_append_c = torch.cat((b,c),dim=0)
print("b_append_c=",b_append_c)
#将列表,数组类型转换成Tensor
num = np.array((1,2,3,4,5))
#注意此处的函数tensor与上面的Tensor函数完全不同
num_tensor1 = torch.tensor(num)
num_tensor2 = torch.from_numpy(num)
print("num_tensor1=",num_tensor1)
print("num_tensor2=",num_tensor2)

#tensor->numpy
b_tensor_np = b.numpy()
print("b_tensor_np=",b_tensor_np)