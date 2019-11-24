import torch

print(torch.__version__)
#
# x = torch.empty(5, 3)
# print(x)
#
#
# x = torch.rand(5, 3)
# print(x)
#
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x)

x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
print(x)

print(x.size())
print(x.shape)

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# adds x to y
y.add_(x)
print(y)


y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())

x += 1
print(x)
print(y) # 也加了1

x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)


x = torch.randn(1)
print(x)
print(x.item())
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True


a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)


import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

c = torch.tensor(a)
a += 1
print(a, c)


# 以下代码只有在PyTorch GPU版本上才会执行
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # GPU
#     y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
#     x = x.to(device)                       # 等价于 .to("cuda")
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型
#

##

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)
print(x.is_leaf, y.is_leaf) # True False

a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)
print(x.grad)
