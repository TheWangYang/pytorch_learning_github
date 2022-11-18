# 5.1 PyTorch模型定义的方式


# PyTorch中定义模型主要是从两个方面入手：模型初始化__init__()和模型前向传播过程forward()；
# 基于nn.Moudle，可以使用三种不同的方式定义模型：Sequential，ModuleList和ModuleDict；
# 1.Sequential类：当模型只是通过简单的叠加层实现时，可以使用该方法实现更加简单，其可以接收一个有序字典OrderedDict
# 或者一系列子模块作为参数来构建Moudule对象；
# Sequential方式创建模型代码：
import torch


class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
            else:
                for idx, module in enumerate(args):
                    self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)

        return input


# Sequential的直接排列：
import torch.nn as nn
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)


print(net)



# 使用OrderedDict搭建模型：
import collections
import torch.nn as nn
net2 = nn.Sequential(
    collections.OrderedDict(
        ['fc1', nn.Linear(784, 256),
         ('relu', nn.ReLU()),
         ('fc2', nn.Linear(256, 10))
         ]
    )
)

print(net2)




# 同时，使用Sequential定义模型不需要定义forward()，因为前向传播的顺序已经定义好了
# 但是使用Sequential方法不灵活，向模型中加一个额外的输入时，会很不方便；




# 使用ModuleList方法
# ModuleList接受一个模型列表作为输入，也可以类似List那样进行append/extend操作；
# 子模块和层的权重也会加入到模型中；
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net[-1])
print(net)




# 需要特别注意的是，nn.ModuleList并没有定义一个模型，而是将不同的模块储存在一起；
# 同时，ModuleList中的模块顺序并不是网络中的真实顺序，需要使用forward()函数指定顺序即可；

class model(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.moduleList = list()

    def forward(self, x):
        for layer in self.moduleList:
            xx = layer(x)
        return x




# 使用ModuleDict定义模型，和第二种的ModuleList相似，就是该种方法可以较为方便地添加模型层的名称：
net = nn.ModuleDict(
    {
        'linear': nn.Linear(784, 256),
        'act': nn.ReLU(),
    }
)


net ['output'] = nn.Linear(256, 10)
print(net['linear'])
print(net.output)
print(net)



# 三种方法的使用场景分析：
# ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；
#
# 当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。




# 5.2 使用模型快速搭建复杂网络

# 组成U-Net的模型块主要有如下几个部分：
#
# 1）每个子块内部的两次卷积（Double Convolution）
#
# 2）左侧模型块之间的下采样连接，即最大池化（Max pooling）
#
# 3）右侧模型块之间的上采样连接（Up sampling）
#
# 4）输出层的处理


# 除模型块外，还有模型块之间的横向连接，输入和U-Net底部的连接等计算，这些单独的操作可以通过forward函数来实现。



# U-Net的pytorch实现：
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super.__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)






class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super.__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # 在维度1上进行拼接
        return self.conv(x)





# 使用上述定义好的模型快组装U-Net
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64,128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)

        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




# 5.3 PyTorch修改模型
# 对torchvision中的ResNet50添加一个fc层：
from collections import OrderedDict
classifier = nn.Sequential(
    OrderedDict([('fc1', nn.Linear(2048, 128)),
                 ('relu', nn.ReLU()),
                 ('dropout', nn.Dropout(0.5)),
                 ('fc2', nn.Linear*(128, 10)),
                 ('output', nn.Softmax(dim=1))
                 ])
)


# 相当于将net最后的fc层替换为classifier结构
net.fc = classifier





# 5.3.2 添加外部输入
# 基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改。
# 具体代码如下：
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias = True)
        self.output = nn.Softmax(dim=1)



    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)),
                      add_variable.unsqueeze(1)), 1)
        x = self.fc_add(x)
        x = self.output(x)
        return x




# 这里的实现要点是通过torch.cat实现了tensor的拼接。torchvision中的resnet50输出是一个1000维的tensor，我们通过修改forward函数（配套定义一些层），先将1000维的tensor通过激活函数层和dropout层，再和外部输入变量"add_variable"拼接，最后通过全连接层映射到指定的输出维度10。
# 另外这里对外部输入变量"add_variable"进行unsqueeze操作是为了和net输出的tensor保持维度一致，常用于add_variable是单一数值 (scalar) 的情况，此时add_variable的维度是 (batch_size, )，需要在第二维补充维数1，从而可以和tensor进行torch.cat操作。



# 5.3.3 添加额外输出
# 有时候在模型训练中，除了模型最后的输出外，我们需要输出模型某一中间层的结果，以施加额外的监督，获得更好的中间层结果。基本的思路是修改模型定义中forward函数的return变量。

class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)



    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000




# 5.4 PyTorch模型保存与读取
# pytorch主要采用pkl,pt,pth三种格式进行模型的保存；
# 一个模型包含：模型结构和权重；
# 其中，模型是继承自nn.Module的类，权重是保存在字典中（key是层名，value是权重向量）；
# 因此，保存模型也有两种形式：保存模型结构和权重/只保存模型权重；
# 详细代码如下：
from collections import models
model = models.resnet152(pretrained=True)


# 保存模型结构和权重
torch.save(model, save_dir)
# 只保存模型权重
torch.save(model.state_dict, save_dir)




# 5.4.3 单卡和多卡模型存储的区别
# pytorch中将模型和数据放到GPU上有两种形式：.cuda()和.to(device)
# 如果使用多卡训练的话，需要对模型使用torch.nn.DataParallel
# 多卡训练代码实示例如下：
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = model.cuda()
model = torch.nn.DataParallel(model).cuda()


# 5.4.4 对模型保存和加载的四种可能情况进行分类讨论
# 1.单卡保存+单卡加载

import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = models.resnet152(pretrained=True)
model.cuda()



# 保存模型 + 读取模型
torch.save(model, save_dir)
loaded_model = torch.load(save_dir)
loaded_model.cuda()



# 保存 + 读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()
loaded_model.state_dict = loaded_dict
loaded_model.cuda()



# 2.单卡保存 + 多卡加载
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES']='0'
model = models.resnet152(pretrained=True)
model.cuda()

# 保存 + 读取整个模型
torch.save(model, save_dir)

os.environ['CUDA_VISIBLE_DEVICES']='1,2'
loaded_model = torch.load(save_dir)
loaded_model = nn.DataParallel(loaded_model).cuda()


# 保存 + 读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()
loaded_model.state_dict = loaded_dict
loaded_model = nn.DataParallel(loaded_model).cuda()



# 3.多卡保存 + 单卡加载
# 需要重点解决的问题：如何去掉保存在多卡中模型权重字典中的module名，以保证模型的统一性
import os
import torch
from torchvision import models



os.environ['CUDA_VISIBLE_DEVICES']='1,2'



model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()


# 保存 + 读取整个模型
torch.save(model, save_dir)


os.environ['CUDA_VISIBLE_DEVICES']='0'
loaded_model = torch.load(save_dir)
loaded_model = loaded_model.module



# 向model里添加module，这种方法较为简单
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model = nn.DataParallel(loaded_model).cuda()
loaded_model.state_dict = loaded_dict



# 遍历字典，去除module
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号

loaded_dict = torch.load(save_dir)

new_state_dict = OrderedDict()
for k, v in loaded_dict.items():
    name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
    new_state_dict[name] = v #新字典的key值对应的value一一对应

loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = new_state_dict
loaded_model = loaded_model.cuda()



# 使用replace去除module

loaded_model = models.resnet152()
loaded_dict = torch.load(save_dir)
loaded_model.load_state_dict({k.replace('module.', ''): v for k, v in loaded_dict.items()})





# 4.多卡保存 + 多卡加载
# 可能出现的问题：
# 1.读取整个模型再使用nn.DataParallel进行分布式训练设置
#
# 这种情况很可能会造成保存的整个模型中GPU id和读取环境下设置的GPU id不符，训练时数据所在device和模型所在device不一致而报错。
# 2.读取整个模型而不使用nn.DataParallel进行分布式训练设置
#
# 这种情况可能不会报错，测试中发现程序会自动使用设备的前n个GPU进行训练（n是保存的模型使用的GPU个数）。此时如果指定的GPU个数少于n，则会报错。在这种情况下，只有保存模型时环境的device id和读取模型时环境的device id一致，程序才会按照预期在指定的GPU上进行分布式训练。
#


# 综上所述，多卡模式下，建议使用权重的方式存储和读取模型：

import os
import torch
from torchvision import models


os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'


model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()



# 保存 + 读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()
loaded_model = nn.DataParallel(loaded_model).cuda()
loaded_model.state_dict = loaded_dict

# 如果只有保存的整个模型，也可以采用提取权重的方式构建新的模型：
#
# # 读取整个模型
loaded_whole_model = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_whole_model.state_dict
loaded_model = nn.DataParallel(loaded_model).cuda()



# 另外，上面所有对于loaded_model修改权重字典的形式都是通过赋值来实现的，在PyTorch中还可以通过"load_state_dict"函数来实现：
#
loaded_model.load_state_dict(loaded_dict)


















