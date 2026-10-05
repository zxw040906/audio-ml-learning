"""
定义类(存储对象与函数的集合)
定义自己的类(无继承)
利用已存在的类作为模板继承修改自己的类(继承)
"""
class Myclass:
    def __init__(self,name):
        self.name = name
    def call_name(self):
        print(f"This is my class and name is {self.name}")
    def __call__(self):
        print(f"This is magic function and name is {self.name}")

myclass = Myclass(name="zxw's Pytorch!!!!")
myclass.call_name()

myclass()