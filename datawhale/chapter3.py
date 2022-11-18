# 第三章；pytorch的重要组成部分


# 深度学习需要在数据加载中进行充分的设计；需要对一些CNN中使用的结构进行预先搭建，再将这些模块组装起来；
# 损失函数和优化器需要能够保证反向传播可以在自定义的模块上进行实现；



# 下面展示了导入torch包的常用代码
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer

# 如下超参数可以统一进行设置；batch size/初始化学习率learning rate/训练次数max_epochs/GPU配置

# 使用GPU时需要的设置：
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



# PyTorch中进行数据读入是通过Dataset和DataLoader实现的：
# Dataset定义好数据的格式和变换形式，DataLoader使用iterative的方式不断读入batch数据；
# 可以定义自己的Dataset类但是需要继承自pytorch官方提供的Dataset类；
# 其中重点关注的函数有：__init__: 用于向类中传入外部参数，同时定义样本集
#
# __getitem__: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
#
# __len__: 用于返回数据集的样本数
#
# 使用torch的自带类：ImageFolder读取数据的代码如下：
import torch
from torchvision import datasets
train_data = datasets.ImageFolder(train_path, transform=data_transform)
val_data = datasets.ImageFolder(val_path, transform=data_transform)
# 其中datatransform可以自定义对图像进行的变换；
# 其中，对于图片放入到一个文件夹且图片的名称和对应的标签的情况，定义Dataset类如下：
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_file)



# 使用DataLoader类进行读取数据的示例代码如下：
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)


# 上述代码中的参数解释：batch_size：样本是按“批”读入的，batch_size就是每次读入的样本数
#
# num_workers：有多少个进程用于读取数据
#
# shuffle：是否将读入的数据打乱
#
# drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练


# 得到加载的数据代码如下：
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))
print(images.shape)
plt.imshow(images[0].transpose(1,2,0))
plt.show()


# Module是一个模型构造类，是所有卷积神经网络模块的基类；
# 定义多层感知机代码如下：
import torch
from torch import nn


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        o = self.act(self.hidden(x))
        return self.output(o)


# 上述代码中无需定义反向传播，torch会自动求梯度并生成backward函数；
# 下述代码展示了使用MLP类定义模型变量net，并对输入数据进行正向传播：
# 其中，net(x)会调用MLP类继承自Module类的call函数，这个函数将被调用forward函数来完成前向计算：
X = torch.rand(2,784)
net = MLP()
print(net)
net(X)


# Module类的子类可以是一个层也可以是一个模型，或者是模型的一个部分；

# parameters类是Tensor的子类，如果一个tensor是一个parameters类，那么它将会被自动添加到模型的参数列表中，
# 除了将参数定义为tensor类之外，还可以将参数定义为：ParameterList和ParameterDict类变量；
# 下面为使用上述list和dict两种方法进行参数列表的定义和反向传播代码：
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyListDense()
print(net)
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)


# 下面介绍了使用torch定义卷积操作和定义卷积核的代码：
import torch
from torch import nn

# 卷积运算（二维互相关）
def corr2d(X, K):
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 使用定义的模块进行测试的代码和测试结果：



import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维:批量和通道


# 注意这里是两侧分别填充1⾏或列，所以在两侧一共填充2⾏或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,padding=1)

X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape
torch.Size([8, 8])


# 技巧：使用长和宽不同的卷积核，也可以实现x输入和输出保持一样：
# 当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽。

# 使用高为5、宽为3的卷积核。在⾼和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

# 对stride步长的介绍：
# 在二维互相关运算中，卷积窗口从输入数组的最左上方开始，按从左往右、从上往下 的顺序，依次在输⼊数组上滑动。我们将每次滑动的行数和列数称为步幅(stride)。

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
torch.Size([2, 2])



# 此外，还可以使用torch.nn包来构建神经网络。我们已经介绍了autograd包，nn包则依赖于autograd包来定义模型并对它们求导。
# 一个nn.Module包含各个层和一个forward(input)方法，该方法返回output。

# 一个神经网络的典型训练过程如下：
#
# 定义包含一些可学习参数(或者叫权重）的神经网络
#
# 在输入数据集上迭代
#
# 通过网络处理输入
#
# 计算 loss (输出和正确答案的距离）
#
# 将梯度反向传播给网络的参数
#
# 更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient


# 使用上述知识点定义LeNet网络

import torch
import torch.nn as nn
import torch.nn.fuctional as F

class LeNet(nn.Module):
    # 设置初始化参数
    def __init__(self):
        super(LeNet, slef).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 设置计算输入和权重相互关系的操作
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # 设置一个max pooling层
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵，那么后边的pooling区域可以定义为一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 做展平处理
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



    def num_flat_features(self, x):
        size = x.size()[1:]  # 除掉批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net = Net()
print(net)


# 上述定义的是LeNet网络torch代码


# 重点：我们只需要定义 forward 函数，backward函数会在使用autograd时自动定义，backward函数用来计算导数。我们可以在 forward 函数中使用任何针对张量的操作和计算。

# 一个模型的可学习参数可以使用net.parameters()返回

# 注意：torch.nn只支持小批量的输入，不支持单个tensor的输入，如果是一个单独的样本，只需要使用input.unsqueeze(0) 来添加一个“假的”批大小维度。


# torch.Tensor - 一个多维数组，支持诸如backward()等的自动求导操作，同时也保存了张量的梯度。
#
# nn.Module - 神经网络模块。是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能。
#
# nn.Parameter - 张量的一种，当它作为一个属性分配给一个Module时，它会被自动注册为一个参数。
#
# autograd.Function - 实现了自动求导前向和反向传播的定义，每个Tensor至少创建一个Function节点，该节点连接到创建Tensor的函数并对其历史进行编码。


# AlexNet网络的搭建代码：


class AlexNet(nn.Module):
    super(AlexNet, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(1, 96, 11, 4),
        nn.ReLu(),
        nn.MaxPool2d(3, 2),
        # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
        # 前两个卷积层后不使用池化层来减小输入的高和宽
        nn.Conv2d(256, 384, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(384, 384, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(384, 256, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(3, 2)
    )

    # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
    self.fc = nn.Sequential(
        nn.Linear(256 * 5 * 5, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10),
    )

    def forward(self, img):
        feature = self.conv(img)
        # 将samples数量维度作为标准，然后将每个图像的三个维度（长宽高）进行展平处理之后，
        # 输入到fc层中
        output = self.fc(feature.view(img.shape[0], -1))
        return output



# 上述定义的为AlexNet网络架构



# pytorch.nn.init()函数中提供了一些对模型的初始化方法

# 通过访问torch.nn.init的官方文档链接 ，我们发现torch.nn.init提供了以下初始化方法：
# 1 . torch.nn.init.uniform_(tensor, a=0.0, b=1.0)
# 2 . torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
# 3 . torch.nn.init.constant_(tensor, val)
# 4 . torch.nn.init.ones_(tensor)
# 5 . torch.nn.init.zeros_(tensor)
# 6 . torch.nn.init.eye_(tensor)
# 7 . torch.nn.init.dirac_(tensor, groups=1)
# 8 . torch.nn.init.xavier_uniform_(tensor, gain=1.0)
# 9 . torch.nn.init.xavier_normal_(tensor, gain=1.0)
# 10 . torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan__in', nonlinearity='leaky_relu')
# 11 . torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
# 12 . torch.nn.init.orthogonal_(tensor, gain=1)
# 13 . torch.nn.init.sparse_(tensor, sparsity, std=0.01)
# 14 . torch.nn.init.calculate_gain(nonlinearity, param=None)



# 我们可以发现这些函数除了calculate_gain，所有函数的后缀都带有下划线，意味着这些函数将会直接原地更改输入张量的值。



# 通常，我们会根据实际模型来初始化其中的参数：
# 使用isinstance来判断模块是什么类型，
isinstance(conv, nn.Conv2d)

# 然后对于不同每个类的层，我们采用不同的初始化参数方法来进行初始化

# 通常，会将各种初始化方法定义为一个initialize_weights()函数进行封装：

def initialize_weights(self):
	for m in self.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zeros_()






# 优秀的loss function可以对模型的优化起到很好的效果：
# 下面是pytorch中包含的Loss Function
# 1.二分类交叉熵loss：torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# 其中的参数表示的含义：weight:每个类别的loss设置权值
#
# size_average:数据为bool，为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
#
# reduce:数据类型为bool，为True时，loss的返回是标量。
# 代码示例：m = nn.Sigmoid()
# loss = nn.BCELoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# output = loss(m(input), target)
# output.backward()


# 2.交叉熵损失函数
# torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# 代码示例：
loss  = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

print(output)


# 3.L1损失函数
# 功能：计算输出y和真实标签target之间的差值的绝对值；
# 我们需要知道的是，reduction参数决定了计算模式。有三种计算模式可选：none：逐个元素计算。 sum：所有元素求和，返回标量。 mean：加权平均，返回标量。 如果选择none，
# 那么返回的结果是和输入元素相同尺寸的。默认计算方式是求平均。
# loss = nn.L1Loss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5)
# output = loss(input, target)
# output.backward()
# print('L1损失函数的计算结果为',output)


# 4.MSE loss函数
# torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
#  计算输出y和真实标签target之差的平方；
# 和L1Loss一样，MSELoss损失函数中，reduction参数决定了计算模式。有三种计算模式可选：none：逐个元素计算。 sum：所有元素求和，返回标量。默认计算方式是求平均；
# 示例代码如下：
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
print('MSE损失函数的计算结果为',output)

# 5.平滑L1 loss函数
# L1的平滑输出，其功能是减轻离群点带来的影响

loss = nn.SmoothL1Loss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()



# Smooth L1和L1对比：对于smoothL1来说，在 0 这个尖端处，过渡更为平滑。



# 6.目标泊松分布的负对数似然损失：泊松分布的负对数似然损失函数
# torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
# 参数：log_input：输入是否为对数形式，决定计算公式，具体公式可以另行参考资料；
#
# full：计算所有 loss，默认为 False。
#
# eps：修正项，避免 input 为 0 时，log(input) 为 nan 的情况。


# 示例代码：

loss = nn.PoissonNLLLoss()
log_input = torch.randn(5, 2, requires_grad=True)
target = torch.randn(5, 2)
output = loss(log_input, target)
output.backward()


# 7.KL散度
# torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
# 计算KL散度，也就是计算相对熵。用于连续分布的距离度量，并且对离散采用的连续输出空间分布进行回归通常很有用。
# 参数设置：none：逐个元素计算。
#
# sum：所有元素求和，返回标量。
#
# mean：加权平均，返回标量。
#
# batchmean：batchsize 维度求平均值。


# 示例代码：
inputs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2]], dtype=torch.float)
loss = nn.KLDivLoss()
output = loss(inputs,target)

print('KLDivLoss损失函数的计算结果为', output)




# 8.MarginRankingLoss
# 计算两个向量之间的相似度，用于排序任务。该方法用于计算两组数据之间的差异。
# 参数设置：margin：边界值x1与x2之间的差异值。
# reduction：计算模式，可为 none/sum/mean。
# 注意：计算公式参考其他资料
# 示例代码：

loss = nn.MarginRankingLoss()
input1 = torch.randn(3, requires_grad=True)
input2 = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()
output = loss(input1, input2, target)
output.backward()

print('MarginRankingLoss损失函数的计算结果为',output)



# 9.多标签边界loss函数

# torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')

# 参数设置：reduction：计算模式，可为 none/sum/mean。

# 计算公式请参考其他资料；

# 示例代码：

loss = nn.MultiLabelMarginLoss()
x = torch.FloatTensor([[0.9, 0.2, 0.4, 0.8]])
# for target y, only consider labels 3 and 0, not after label -1
y = torch.LongTensor([[3, 0, -1, 1]])# 真实的分类是，第3类和第0类
output = loss(x, y)

print('MultiLabelMarginLoss损失函数的计算结果为',output)
MultiLabelMarginLoss损失函数的计算结果为 tensor(0.4500)





# 10.二分类损失函数
# torch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')torch.nn.(size_average=None, reduce=None, reduction='mean')
# 功能：计算logistic loss
# 计算公式见其他参考资料；
# 示例代码：

inputs = torch.tensor([[0.3, 0.7], [0.5, 0.5]])  # 两个样本，两个神经元
target = torch.tensor([[-1, 1], [1, -1]], dtype=torch.float)  # 该 loss 为逐个神经元计算，需要为每个神经元单独设置标签

loss_f = nn.SoftMarginLoss()
output = loss_f(inputs, target)

print('SoftMarginLoss损失函数的计算结果为',output)
SoftMarginLoss损失函数的计算结果为 tensor(0.6764)






# 11.多分类折页损失
# torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
# 计算多分类的折页损失；
# 参数设置：reduction：计算模式，可为 none/sum/mean。
#
# p：可选 1 或 2。
#
# weight：各类别的 loss 设置权值。
#
# margin：边界值

# 计算公式见其他资料；
# 示例代码：
inputs = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
target = torch.tensor([0, 1], dtype=torch.long)

loss_f = nn.MultiMarginLoss()
output = loss_f(inputs, target)

print('MultiMarginLoss损失函数的计算结果为',output)
MultiMarginLoss损失函数的计算结果为 tensor(0.6000)







# 12.三元组损失
# 功能：三元组损失
# 三元组: 这是一种数据的存储或者使用格式。<实体1，关系，实体2>。在项目中，也可以表示为< anchor, positive examples , negative examples>
#
# 在这个损失函数中，我们希望去anchor的距离更接近positive examples，而远离negative examples
#
# 主要参数:
#
# reduction：计算模式，可为 none/sum/mean。
#
# p：可选 1 或 2。
#
# margin：边界值


# 计算公式见其他资料；

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = triplet_loss(anchor, positive, negative)
output.backward()
print('TripletMarginLoss损失函数的计算结果为',output)
TripletMarginLoss损失函数的计算结果为 tensor(1.1667, grad_fn=<MeanBackward0>)





# 13.HingEmbeddingLoss
# torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
# 功能：对输出的embedding结果做Hing损失计算
# 主要参数:
#
# reduction：计算模式，可为 none/sum/mean。
#
# margin：边界值

# 计算公式：
loss_f = nn.HingeEmbeddingLoss()
inputs = torch.tensor([[1., 0.8, 0.5]])
target = torch.tensor([[1, 1, -1]])
output = loss_f(inputs,target)

print('HingEmbeddingLoss损失函数的计算结果为',output)
# HingEmbeddingLoss损失函数的计算结果为 tensor(0.7667)






# 14.余弦相似度
# torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
# 功能：对两个向量做余弦相似度；
# 主要参数:
#
# reduction：计算模式，可为 none/sum/mean。
#
# margin：可取值[-1,1] ，推荐为[0,0.5] 。
# 这个损失函数应该是最广为人知的。对于两个向量，做余弦相似度。将余弦相似度作为一个距离的计算方式，如果两个向量的距离近，则损失函数值小，反之亦然。
# 计算公式见其他参考资料；
# 示例代码：

loss_f = nn.CosineEmbeddingLoss()
inputs_1 = torch.tensor([[0.3, 0.5, 0.7], [0.3, 0.5, 0.7]])
inputs_2 = torch.tensor([[0.1, 0.3, 0.5], [0.1, 0.3, 0.5]])
target = torch.tensor([[1, -1]], dtype=torch.float)
output = loss_f(inputs_1,inputs_2,target)

print('CosineEmbeddingLoss损失函数的计算结果为',output)







# 15.CTC损失函数
# torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
# 功能：用于解决时序类数据的分类
# 计算连续时间序列和目标序列之间的损失。CTCLoss对输入和目标的可能排列的概率进行求和，产生一个损失值，这个损失值对每个输入节点来说是可分的。输入与目标的对齐方式被假定为 "多对一"，这就限制了目标序列的长度，使其必须是≤输入长度。
# 主要参数：reduction：计算模式，可为 none/sum/mean。
#
# blank：blank label。
#
# zero_infinity：无穷大的值或梯度值为


# 示例代码：
# Target are to be padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()


# Target are to be un-padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()

print('CTCLoss损失函数的计算结果为',loss)








# 3.7 训练和评估


# 需要设置模型状态，使用如下代码：
model.train()
model.eval()


# 在模型训练过程中，需要使用for循环读取数据和标签从DataLoader中；
for data, label in train_loader:
    # 然后，将数据放到GPU上用于后续计算，以.cuda()为例：
    data, label, = data.cuda(), label.cuda()

    # 开始使用当前批次数据做训练时，应当先将优化器的梯度设置为0；
    optimizer.zero_grad()

    # 之后data送入到模型中训练：
    output = model(data)

    # 根据预先定义的criterion计算loss函数：
    loss = criterion(output, label)


    # 将loss反向传播回网络：
    loss.backward()


    # 使用优化器更新模型参数：
    optimizer.step()



# 验证和训练不一致的地方：
# 验证/测试的流程基本与训练过程一致，不同点在于：
#
# 需要预先设置torch.no_grad，以及将model调至eval模式
#
# 不需要将优化器的梯度置零
#
# 不需要将loss反向回传到网络
#
# 不需要更新optimizer


# 一个完整的图像分类过程代码如下：
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(label, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
		print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))




# 一个完整的验证过程代码如下：
def val(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, val_loss))



# 3.8 在完成对模型的训练之后，需要将分类的ROC曲线，卷积神经网络中的kernel，以及train_loss和val_loss
# 曲线进行可视化



# 3.9 PyTorch优化器
# 深度学习的本质就是使用一个函数去寻找最优解，只不过这个最优解是一个矩阵，而如何最快地
# 求得最优解是重点；

# 目前使用最多的就是：BP + 优化器求解；

# PyTorch中提供的优化器：
# torch.optim.ASGD
#
# torch.optim.Adadelta
#
# torch.optim.Adagrad
#
# torch.optim.Adam
#
# torch.optim.AdamW
#
# torch.optim.Adamax
#
# torch.optim.LBFGS
#
# torch.optim.RMSprop
#
# torch.optim.Rprop
#
# torch.optim.SGD
#
# torch.optim.SparseAdam



# 在torch.optim中提供了10种优化器：
# 而上述所有优化器都是继承自Optimizer：

class Optimizer(object):
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []


# 参数详解：defaults：存储的是优化器的超参数，例子如下：
#
# {'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False}
# state：参数的缓存，例子如下：
#
# defaultdict(<class 'dict'>, {tensor([[ 0.3864, -0.0131],
#         [-0.1911, -0.4511]], requires_grad=True): {'momentum_buffer': tensor([[0.0052, 0.0052],
#         [0.0052, 0.0052]])}})
# param_groups：管理的参数组，是一个list，其中每个元素是一个字典，顺序是params，lr，momentum，dampening，weight_decay，nesterov，例子如下：
# [{'params': [tensor([[-0.1022, -1.6890],[-1.5116, -1.7846]], requires_grad=True)], 'lr': 1, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False}]




# zero_grad()：清空所管理参数的梯度，PyTorch的特性是张量的梯度不自动清零，因此每次反向传播后都需要清空梯度。
# 下面是清空参数中梯度的代码：
def zero_grad(self, set_to_none: bool = False):
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is not None:  #梯度不为空
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()# 梯度设置为0



# 参数解释：step()：执行一步梯度更新，参数更新
def step(self, closure):
    raise NotImplementedError


# add_param_group()：添加参数组，代码如下：
def add_param_group(self, param_group):
    assert isinstance(param_group, dict), "param group must be a dict"
# 检查类型是否为tensor
    params = param_group['params']
    if isinstance(params, torch.Tensor):
        param_group['params'] = [params]
    elif isinstance(params, set):
        raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                        'the ordering of tensors in sets will change between runs. Please use a list instead.')
    else:
        param_group['params'] = list(params)
    for param in param_group['params']:
        if not isinstance(param, torch.Tensor):
            raise TypeError("optimizer can only optimize Tensors, "
                            "but one of the params is " + torch.typename(param))
        if not param.is_leaf:
            raise ValueError("can't optimize a non-leaf Tensor")

    for name, default in self.defaults.items():
        if default is required and name not in param_group:
            raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                             name)
        else:
            param_group.setdefault(name, default)

    params = param_group['params']
    if len(params) != len(set(params)):
        warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                      "in future, this will cause an error; "
                      "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)
# 上面好像都在进行一些类的检测，报Warning和Error
    param_set = set()
    for group in self.param_groups:
        param_set.update(set(group['params']))

    if not param_set.isdisjoint(set(param_group['params'])):
        raise ValueError("some parameters appear in more than one parameter group")
# 添加参数
    self.param_groups.append(param_group)



# load_state_dict() ：加载状态参数字典，可以用来进行模型的断点续训练，继续上次的参数进行训练
def load_state_dict(self, state_dict):
    r"""Loads the optimizer state.

    Arguments:
        state_dict (dict): optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
    """
    # deepcopy, to be consistent with module API
    state_dict = deepcopy(state_dict)
    # Validate the state_dict
    groups = self.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
              zip(chain.from_iterable((g['params'] for g in saved_groups)),
                  chain.from_iterable((g['params'] for g in groups)))}

    def cast(param, value):
        r"""Make a deep copy of value, casting all tensors to device of param."""
   		.....

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
       ...
    param_groups = [
        update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    self.__setstate__({'state': state, 'param_groups': param_groups})






# state_dict()：获取优化器当前状态信息字典，代码如下：
def state_dict(self):
    r"""Returns the state of the optimizer as a :class:`dict`.

    It contains two entries:

    * state - a dict holding current optimization state. Its content
        differs between optimizer classes.
    * param_groups - a dict containing all parameter groups
    """
    # Save order indices instead of Tensors
    param_mappings = {}
    start_index = 0

    def pack_group(group):
		......
    param_groups = [pack_group(g) for g in self.param_groups]
    # Remap state to use order indices as keys
    packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                    for k, v in self.state.items()}
    return {
        'state': packed_state,
        'param_groups': param_groups,
    }






# 3.9.2 实际操作，代码如下：
# 使用上述公式进行实际操作的代码如下：
import os
import torch

# 设置权重，服从正态分布  --> 2 x 2
weight = torch.randn((2, 2), requires_grad=True)
# 设置梯度为全1矩阵  --> 2 x 2
weight.grad = torch.ones((2, 2))
# 输出现有的weight和data
print("The data of weight before step:\n{}".format(weight.data))
print("The grad of weight before step:\n{}".format(weight.grad))
# 实例化优化器
optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
# 进行一步操作
optimizer.step()
# 查看进行一步后的值，梯度
print("The data of weight after step:\n{}".format(weight.data))
print("The grad of weight after step:\n{}".format(weight.grad))
# 权重的梯度清零
optimizer.zero_grad()
# 检验权重的梯度是否为0
print("The grad of weight after optimizer.zero_grad():\n{}".format(weight.grad))
# 输出参数
print("optimizer.params_group is \n{}".format(optimizer.param_groups))
# 查看参数位置，optimizer和weight的位置一样，我觉得这里可以参考Python是基于值管理
print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))
# 添加参数：weight2
weight2 = torch.randn((3, 3), requires_grad=True)
optimizer.add_param_group({"params": weight2, 'lr': 0.0001, 'nesterov': True})
# 查看现有的参数
print("optimizer.param_groups is {}.\n".format(optimizer.param_groups))
# 查看当前状态信息
opt_state_dict = optimizer.state_dict()
print("state_dict before step:{}.\n", opt_state_dict)
# 进行5次step操作
for _ in range(50):
    optimizer.step()
# 输出现有状态信息
print("state_dict after step:{}.\n", optimizer.state_dict())
# 保存参数信息
torch.save(optimizer.state_dict(),os.path.join(r"D:\pythonProject\Attention_Unet", "optimizer_state_dict.pkl"))
print("----------done-----------")
# 加载参数信息
state_dict = torch.load(r"D:\pythonProject\Attention_Unet\optimizer_state_dict.pkl") # 需要修改为你自己的路径
optimizer.load_state_dict(state_dict)
print("load state_dict successfully\n{}".format(state_dict))
# 输出最后属性信息
print("\n{}".format(optimizer.defaults))
print("\n{}".format(optimizer.state))
print("\n{}".format(optimizer.param_groups))





# 3.9.3 输出结果
# 进行更新前的数据，梯度
The
data
of
weight
before
step:
tensor([[-0.3077, -0.1808],
        [-0.7462, -1.5556]])
The
grad
of
weight
before
step:
tensor([[1., 1.],
        [1., 1.]])
# 进行更新后的数据，梯度
The
data
of
weight
after
step:
tensor([[-0.4077, -0.2808],
        [-0.8462, -1.6556]])
The
grad
of
weight
after
step:
tensor([[1., 1.],
        [1., 1.]])
# 进行梯度清零的梯度
The
grad
of
weight
after
optimizer.zero_grad():
tensor([[0., 0.],
        [0., 0.]])
# 输出信息
optimizer.params_group is
[{'params': [tensor([[-0.4077, -0.2808],
                     [-0.8462, -1.6556]], requires_grad=True)], 'lr': 0.1, 'momentum': 0.9, 'dampening': 0,
  'weight_decay': 0, 'nesterov': False}]

# 证明了优化器的和weight的储存是在一个地方，Python基于值管理
weight in optimizer: 1841923407424
weight in weight: 1841923407424

# 输出参数
optimizer.param_groups is
[{'params': [tensor([[-0.4077, -0.2808],
                     [-0.8462, -1.6556]], requires_grad=True)], 'lr': 0.1, 'momentum': 0.9, 'dampening': 0,
  'weight_decay': 0, 'nesterov': False}, {'params': [tensor([[0.4539, -2.1901, -0.6662],
                                                             [0.6630, -1.5178, -0.8708],
                                                             [-2.0222, 1.4573, 0.8657]], requires_grad=True)],
                                          'lr': 0.0001, 'nesterov': True, 'momentum': 0.9, 'dampening': 0,
                                          'weight_decay': 0}]

# 进行更新前的参数查看，用state_dict
state_dict
before
step:
{'state': {0: {'momentum_buffer': tensor([[1., 1.],
                                          [1., 1.]])}},
 'param_groups': [{'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0]},
                  {'lr': 0.0001, 'nesterov': True, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'params': [1]}]}
# 进行更新后的参数查看，用state_dict
state_dict
after
step:
{'state': {0: {'momentum_buffer': tensor([[0.0052, 0.0052],
                                          [0.0052, 0.0052]])}},
 'param_groups': [{'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0]},
                  {'lr': 0.0001, 'nesterov': True, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'params': [1]}]}

# 存储信息完毕
----------done - ----------
# 加载参数信息成功
load
state_dict
successfully
# 加载参数信息
{'state': {0: {'momentum_buffer': tensor([[0.0052, 0.0052],
                                          [0.0052, 0.0052]])}},
 'param_groups': [{'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0]},
                  {'lr': 0.0001, 'nesterov': True, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'params': [1]}]}

# defaults的属性输出
{'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False}

# state属性输出
defaultdict( <


class 'dict'>, {tensor([[-1.3031, -1.1761],
[-1.7415, -2.5510]], requires_grad=True): {'momentum_buffer': tensor([[0.0052, 0.0052],
                                                                      [0.0052, 0.0052]])}

})

# param_groups属性输出
[{'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False,
  'params': [tensor([[-1.3031, -1.1761],
                     [-1.7415, -2.5510]], requires_grad=True)]},
 {'lr': 0.0001, 'nesterov': True, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0,
  'params': [tensor([[0.4539, -2.1901, -0.6662],
                     [0.6630, -1.5178, -0.8708],
                     [-2.0222, 1.4573, 0.8657]], requires_grad=True)]}]



# 重点注意：
# 1.每个优化器是一个类，一定要进行实例化之后才可以使用；
# 2.optimizer需要在CNN训练中的单个epoch中实现两个步骤：梯度设置为0，梯度更新；代码如下：
optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
for epoch in range(EPOCH):
	...
	optimizer.zero_grad()  #梯度置零
	loss = ...             #计算loss
	loss.backward()        #BP反向传播
	optimizer.step()       #梯度更新
# 3.给网络的不同的层设置不同优化器参数：
from torch import optim
from torchvision.models import resnet18

net = resnet18()

optimizer = optim.SGD([
    {'params':net.fc.parameters()},#fc的lr使用默认的1e-5
    {'params':net.layer4[0].conv1.parameters(),'lr':1e-2}],lr=1e-5)

# 可以使用param_groups查看属性

# 3.9.4 实验
# 为了更好的帮大家了解优化器，我们对PyTorch中的优化器进行了一个小测试
#
# 数据生成：
#
# a = torch.linspace(-1, 1, 1000)
# # 升维操作
# x = torch.unsqueeze(a, dim=1)
# y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))




# 使用的网络结构：
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(1, 20)
#         self.predict = nn.Linear(20, 1)
#
#     def forward(self, x):
#         x = self.hidden(x)
#         x = F.relu(x)
#         x = self.predict(x)
#         return x

# 优化器的好坏是根据模型进行改变测试的，不存在非常好的优化器适合所有的网络；






