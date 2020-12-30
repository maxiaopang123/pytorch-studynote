# pytorch安装

https://blog.csdn.net/qq_38704904/article/details/95192856

pytorh:上手简单，动态图调试？？
安装pytorch和torchvision包

输入命令检查包


```python
conda list
```

![1609233934176](C:\Users\marryao\AppData\Roaming\Typora\typora-user-images\1609233934176.png)



# pytorch基础知识

## 1.张量Tensor

可以理解为标量是零阶张量，矢量是一阶张量，矩阵是二阶张量
Tensor可以运行在GPU中，以加速运算
构建一个5×3的随机初始化矩阵，类型dtype是长整型float：


```python
import torch 

x= torch.randn(5,3,dtype=torch.float)#五行三列，float类型矩阵
print(x)
```

    tensor([[ 1.3839,  1.3148, -0.0049],
            [-0.3909, -0.8380,  0.1586],
            [ 1.0122,  1.2690,  1.5543],
            [-2.7078,  0.2349,  1.3415],
            [-1.2921, -0.8683, -0.3582]])


直接用已有数据构建Tensor：


```python
x = torch.tensor([0,1,2,3,4])
print(x)
```

    tensor([0, 1, 2, 3, 4])


tensor可以进行多种运算，以加法为例
语法1：提供一个输出tensor作为参数：


```python
x=torch.randn(5,3)
y=torch.randn(5,3)
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
```

    tensor([[ 1.5697,  2.0826,  3.4048],
            [ 1.5972,  2.4530,  0.4551],
            [-0.4288,  1.7134, -0.0257],
            [ 1.2222,  1.6654,  1.0342],
            [ 1.6193, -2.2244, -0.6156]])


语法2：替换：


```python
#adds x to y
y.add_(x)
print(y)
```

    tensor([[ 1.5697,  5.0826,  9.4048],
            [ 1.5972,  5.4530,  6.4551],
            [-0.4288,  4.7134,  5.9743],
            [ 1.2222,  4.6654,  7.0342],
            [ 1.6193,  0.7756,  5.3844]])


**:任何操作符都固定地在前面加上_来表示替换。

Tensor和Numpy可以想换转换
Tensor转换为Numpy： 


```python
a=torch.tensor([0,1,2,3,4])
print(a)
```

    tensor([0, 1, 2, 3, 4])



```python
b=a.numpy()
print(b)
```

    [0 1 2 3 4]


numpy转化为tensor


```python
import numpy as np
a=np.ones(5)
print(a)
```

    [1. 1. 1. 1. 1.]



```python

b= torch.from_numpy(a)
print(b)
```

    tensor([1., 1., 1., 1., 1.], dtype=torch.float64)


Tensor可以被移动到任何设备中，例如GPU加速运算，使用.to方法即可


```python
x=torch.ones(5)
if torch.cuda.is_available():
    device=torch.device("cuda")
    y=torch.ones_like(x,device=device)  #directly create a tensor on GPU
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))
    
```

并没有输出，问题出在哪里？？？

## 2.自动求导Autograd

Autograd为所有tensor上的操作提供了自动微分机制。这是一个动态运行的机制，即当代码运行的时候，反向传播过程就已经被定义了，且每迭代都会有动态变化。如果把Tensor的属性.requires_grad设置为True，就会追踪它所有的操作。完成计算后，可以通过.backward()来自动完成所有梯度计算


1.创建一个Tensor，属性.requires_grad设置为True，跟踪整个过程。


```python
x = torch.ones(2,2,requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)



```python
#在tensor上进行运算操作：
y=x+1
z=y*y*2
out=z.mean()
print(z,out)
```

    tensor([[8., 8.],
            [8., 8.]], grad_fn=<MulBackward0>) tensor(8., grad_fn=<MeanBackward0>)


out是标量，所以out.backward()等价于out.backward(torch.tensor(1))
接下来是反向传播：


```python
out.backward()
print(x.grad)
```

```python
tensor([[2., 2.],
        [2., 2.]])
```



