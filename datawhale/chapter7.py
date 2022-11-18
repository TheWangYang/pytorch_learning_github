
# 7.1 可视化网络结构

# 使用torchinfo来得到模型的信息
# 7.1.1 使用print函数打印模型信息

# import torchvision.models as models
# model = models.resnet18()
#
# print(model)




# 使用print()只能得到模型的基础构建信息，不能显示出模型每层的shape也不能显示对应参数量的大小

# torchinfo的使用：
# import torchvision.models as models
# from torchinfo import summary
# resnet18 = models.resnet18()
# summary(resnet18, (1, 3, 224, 224))

# 我们可以看到torchinfo提供了更加详细的信息，包括模块信息（每一层的类型、输出shape和参数量）、模型整体的参数量、模型大小、一次前向或者反向传播需要的内存大小等





# 7.2 CNN可视化
# 7.2.1 CNN卷积核可视化
# 可视化卷积核就等价于可视化对应的权重矩阵
# 示例代码如下：

import torch



