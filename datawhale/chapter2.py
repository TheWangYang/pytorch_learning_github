import torch

# x = torch.rand(4, 3)
# print(x)
#
#
# y = torch.rand(4, 3)
#
#
# print(y)
#
#
# #
# # print(x + y)
# #
# # print(x.add_(y))
# #
# # print(torch.add(x, y))

#
# x = torch.rand(4, 3)
#
# # print(x)
# #
# # print(x[:, 2])
#
# y = x.view(12)  # view()函数只更改观察张量的角度，和x贡献内存，更改y也会更改x
#
# print(x.size(), y.size())
#
# z = x.view(-1, 6)
# print(z.size())


# 推荐首先使用.clone()对源张量进行复制，然后对复制后的张量使用view()函数进行处理，此外，复制的张量中grad梯度发生变化时，
# 源张量也会发生改变

# x = torch.rand(3, 4)
# print(x)
# # print(x.item())  # 只能对包含有一个元素的张量tensor使用.item()得到其值，同时，获得其属性
# copy_x = x.clone()
# copy_x.view(-1, 6)
# copy_x += 1
#
# print(x)
# print(copy_x)


# # 测试item()函数的作用
# y = torch.rand(1)
# print(y.item())

#
#
# # 测试tensor的广播机制
# x = torch.arange(1, 3).view(1, 2)
# y = torch.arange(1, 4).view(3, 1)
#
# print(x)
#
# print(y)
#
# print(x + y)


# # 测试requires_grad属性
# x = torch.ones(2, 2, requires_grad=True)
#
# # print(x)
#
# y = x**2
# # print(y)
#
#
# # Because the y is the result of computation, so the y has the grad_fn attribute.
# # print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
#
# #
# # print(z, out)
#
# out.backward()
#
# print(x.grad)
#
# print(x)
#
# out2 = x.sum()
# print(out2)
# out2.backward()
#
# print(x.grad)



# # 雅可比向量积的例子
# x = torch.randn(3, requires_grad=True)
# print(x)
#
# y = x * 2
# i = 0
# while y.data.norm() < 1000:
#     y = y * 2
#     i = i + 1
#
# print(y)
# print(i)
#
# # 这种情况下，由Y不再是标量，因此torch.autograd 不能直接计算完整的雅可比矩阵,
# # 但如果想要雅可比向量积，只需要将这个向量作为参数传给backward
#
# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
#
# y.backward(v)
#
# print(x.grad)




# 可以通过将代码块包装在with torch.no_grad()中，来阻止autograd跟踪设置了.requires_grad=True的张量的历史记录


# print(x.requires_grad)
#
# print((x ** 2).requires_grad)
#
# with torch.no_grad():
#     print((x ** 2).requires_grad)


# 如果想要修改tensor的数值，但是不希望被autograd记录，那么可以对tensor.data进行操作
x = torch.ones(1, requires_grad=True)

print(x.data)
print(x.data.requires_grad)


y = 2 * x
x.data *= 100


y.backward()
print(x)
print(x.grad)


# 使用.cuda()函数实现将代码的运行从cpu中迁移到gpu(0)上
# 常见的并行方法：
# 将网络结构分布到不同的设备中，这种方法对GPU的通信是一个挑战，但是GPU通信在这种任务中很难办到
# 因此，这种方法就淡出了视野；
# 同一层的任务分布到不同的数据中，将模型中的某一层的任务分布到不同GPU中进行计算，
# 但是这种情况下，也会出现和第一种相同的情况；
# 第三种：将数据进行拆分之后，将模型分别放入到多个GPU中进行运算，最后只需要将多个输出数据进行汇总即可；
# 现在主流的方式就是第三种-数据并行的方式；









