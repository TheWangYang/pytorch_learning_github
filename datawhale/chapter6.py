# import os
# import torch
# from torchvision import models
# import torchvision
#
#
#
# # 6.1 自定义损失函数
#
# # 需要通过使用自定义Loss 函数来实现对非通用模型的泛化
# # 6.1.1 以函数的方式定义loss
# def my_loss_func(output, target):
#     loss = torch.mean((output - target) ** 2)
#     return loss
#
#
#
# # 6.1.2 以类方式定义
# # 虽然以函数定义的方式很简单，但是以类方式定义更加常用，在以类方式定义损失函数时，我们如果看每一个损失函数的继承关系我们就可以发现Loss函数部分继承自_loss, 部分继承自_WeightedLoss, 而_WeightedLoss继承自_loss， _loss继承自 nn.Module。我们可以将其当作神经网络的一层来对待，同样地，我们的损失函数类就需要继承自nn.Module类，在下面的例子中我们以DiceLoss为例向大家讲述。
# # Dice Loss是一种在分割领域常见的loss函数，定义如下：
# DSC = (2 * |X 交 Y|) / (|X| + |Y|)
#
# # 上述Loss的实现代码如下：
# class Diceloss(nn.Module):
#     from torch import functional as F
#     def __init__(self, weight=None, size_average=True):
#         super(Diceloss, self).__init__()
#
#
#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice
#
#
#
# criterion = Diceloss()
# loss = criterion(inputs, targets)
#
#
# # 自定义的其他类型Loss
# 除此之外，常见的损失函数还有BCE - Dice
# Loss，Jaccard / Intersection
# over
# Union(IoU)
# Loss，Focal
# Loss......
#
#
# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
#
#         return Dice_BCE
#
#
# --------------------------------------------------------------------
#
#
# class IoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(IoULoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection
#
#         IoU = (intersection + smooth) / (union + smooth)
#
#         return 1 - IoU
#
#
# --------------------------------------------------------------------
#
# ALPHA = 0.8
# GAMMA = 2
#
#
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()
#
#     def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
#
#         return focal_loss
# # 更多的可以参考链接1
#
#
#
#
# # 6.2 动态调整学习率
# # 学习率衰减策略-scheduler
# # 官方提供的衰减API如下：
# # pytorch已经封装的学习率衰减计划为：torch.optim.lr_schedule
#
# # lr_scheduler.LambdaLR
# #
# # lr_scheduler.MultiplicativeLR
# #
# # lr_scheduler.StepLR
# #
# # lr_scheduler.MultiStepLR
# #
# # lr_scheduler.ExponentialLR
# #
# # lr_scheduler.CosineAnnealingLR
# #
# # lr_scheduler.ReduceLROnPlateau
# #
# # lr_scheduler.CyclicLR
# #
# # lr_scheduler.OneCycleLR
# #
# # lr_scheduler.CosineAnnealingWarmRestarts
#
#
# # 使用官方API的方法：
# optimizer = torch.optim.Adam(...)
#
# # 选择上面提到的一种或多种动态调整学习率的方法
# scheduler1 = torch.optim.lr_scheduler....
# scheduler2 = torch.optim.lr_scheduler....
#
# ...
#
#
#
# scheduler = torch.optim.lr_scheduler....
#
# # 进行训练过程如下：
# for epoch in range(100):
#     train()
#     validate()
#     optimizer.step()
#
#     # 需要在优化器参数更新之后，再动态调整学习率
#     scheduler1.step()
#     ...
#     scheduler2.step()
#     schedulern.step()
#
# # 使用注意事项：
# # 我们在使用官方给出的torch.optim.lr_scheduler时，需要将scheduler.step()放在optimizer.step()后面进行使用。
#
#
#
# # 6.2.2 自定义scheduler
#
# def adjust_learning_rate(optimizer, epoch):
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
#
# # 调用上述自定义学习率调整scheduler函数
# optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
#
# for epoch in range(10):
#     train()
#     validate()
#     adjust_learning_rate(optimizer, epoch)
#
#
#
#
#
# # 6.3 模型微调
# # 有时数据集只有几千张，从头开始训练具有几千万参数的大型模型是不现实的，因为会有严重的过拟合现象发生。
# # 使用transfer learning的方法，将从原数据集中学到的知识迁移到现在的目标数据集中。
# # 迁移学习的一个较大的应用场景是：模型微调。
# # 6.3.1 模型微调的流程
# # 1.在源数据集(如ImageNet数据集)上预训练一个神经网络模型，即源模型。
# #
# # 2.创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
# #
# # 3.为目标模型添加一个输出⼤小为⽬标数据集类别个数的输出层，并随机初始化该层的模型参数。
# #
# # 4.在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。
#
#
#
#
# # 6.3.2 使用已有的模型结构
# # 实例化网络：
# import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# denset = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet_v2 = models.mobilenet_v2()
# mobilenet_v3_large = models.mobilenet_v3_large()
# mobilenet_v3_small = models.mobilenet_v3_small()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()
#
#
#
#
#
# # 通过True或者False来决定是否使用预训练好的权重，在默认状态下pretrained = False，意味着我们不使用预训练得到的权重，当pretrained = True，意味着我们将使用在一些数据集上预训练得到的权重。
#
#
# import torchvision.models as models
# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
# mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)
#
#
#
#
#
# # 通常PyTorch模型的扩展为.pt或.pth，程序运行时会首先检查默认路径中是否有已经下载的模型权重，一旦权重被下载，下次加载就不需要下载了。
# #
# # 一般情况下预训练模型的下载会比较慢，我们可以直接通过迅雷或者其他方式去 这里 查看自己的模型里面model_urls，然后手动下载，预训练模型的权重在Linux和Mac的默认下载路径是用户根目录下的.cache文件夹。在Windows下就是C:\Users\<username>\.cache\torch\hub\checkpoint。我们可以通过使用 torch.utils.model_zoo.load_url()设置权重的下载地址。
# #
# # 如果觉得麻烦，还可以将自己的权重下载下来放到同文件夹下，然后再将参数加载网络。
# #
# # self.model = models.resnet50(pretrained=False)
# # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
# # 如果中途强行停止下载的话，一定要去对应路径下将权重文件删除干净，要不然可能会报错。
#
#
#
# # 6.3.3 训练特定层
#
# # 冻结部分参数层的梯度传播
# # 在默认情况下，参数的属性.requires_grad = True，如果我们从头开始训练或微调不需要注意这里。但如果我们正在提取特征并且只想为新初始化的层计算梯度，其他参数不进行改变。那我们就需要通过设置requires_grad = False来冻结部分层。在PyTorch官方中提供了这样一个例程。
#
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
# #             param.requires_grad = False
# #
# #
# #
# # # 首先冻结参数，然后修改最后的FC层权重和参数
# # import torchvision.models as models
# # # 冻结参数的梯度
# # feature_extract = True
# # model = models.resnet18(pretrained=True)
# # set_parameter_requires_grad(model, feature_extract)
# # # 修改模型
# # num_ftrs = model.fc.in_features
# # model.fc = nn.Linear(in_features=num_ftrs, out_features=4, bias=True)
# #
# #
# #
# #
# # # 之后在训练过程中，model仍会进行梯度回传，但是参数更新则只会发生在fc层。通过设定参数的requires_grad属性，我们完成了指定训练模型的特定层的目标，这对实现模型微调非常重要。
#
# # 6.3 模型微调-timm
# # timm是一个较为常见的预训练模型库，里面包含了很多CV领域SOTA的模型。
# # 截止到3.27日为止，timm提供的模型达到了592个，查看其中包含具体模型的方法代码示例如下：
# import timm
#
# avali_pretrained_models = timm.list_models(pretrained=True)
# print(len(avali_pretrained_models))
#
#
# # 查询某种模型的种类
# all_densenet_models = timm.list_models("*densenet*")
# print(all_densenet_models)
#
#
#
# # 查看模型具体参数：
# model = timm.create_model('resnet34', num_classes=10, pretrained=True)
# print(model.default_cfg)
#
#
#
# # 使用timm提供的模型并对预训练模型进行修改
# # 使用timm.create_model()来创建模型
# # 查看模型中某一层的参数：
# model = timm.create_model('resnet34', pretrained=True)
#
#
# list(dict(model.named_children())['conv1'].parameters())
#
#
#
# # 修改模型
# model = timm.create_model('resnet34', num_classes=10, pretrained=True)
# x = torch.randn(1, 3, 224, 224)
# output = model(x)
# output.shape
#
#
#
# # 改变模型的输入通道数：
# model = timm.create_model('resnet34', num_classes=10, pretrained=True)
# x = torch.randn(1, 1, 224, 224)
# output = model(x)
#
#
#
# # 使用timm保存模型
#
# torch.save(model.state_dict(), save_dir)
# model.load_state_dict(torch.load(save_dir))
#
#
#
#
#
# # 6.4 半精度训练
# # 我们观察PyTorch默认的浮点数存储方式用的是torch.float32，小数点后位数更多固然能保证数据的精确性，但绝大多数场景其实并不需要这么精确，只保留一半的信息也不会影响结果，也就是使用torch.float16格式。由于数位减了一半，因此被称为“半精度”，具体如下图：
# # 很显然，使用半精度训练可以减少显存占用，使得显卡可以以较大的batch_size进行训练；
# # 6.4.1 半精度训练的设置
# # 使用autocast配置半精度训练，同时需要设置如下：
# from torch.cuda.amp import autocast
#
# # 模型设置
# # 使用autocast装饰器的方法，来装饰forward()函数
# @autocast
# def forward(self, x):
#     ...
#     return x
#
#
#
#
# # 训练过程
# for x in train_loader:
#     x = x.cuda()
#     with autocast():
#         output = model(x)
#
#
#
#
#
# # 重点注意：半精度训练主要适合用于训练数据本身size较大的情况。
#
#
#
#
#
#
#
# # 6.5 数据增强
# # 该方法可以提高数据集的大小和质量。
# # imgaug库相较于torchvision.transforms，它提供了更多的数据增强方法。
#
#
#

# 6.5.2 imgaug的使用
# imgaug仅仅提供了对图像进行增强的方法，没有提供IO操作，因此需要使用一些库将图像导入，
# 推荐使用imageio进行读取，如果使用opencv的话，需要手动改变通道。
# 除此以外，当我们用PIL.Image进行读取时，因为读取的图片没有shape的属性，所以我们需要将读取到的img转换为np.array()的形式再进行处理。因此官方的例程中也是使用imageio进行图片读取。
# 单张图像处理：
# import imageio
# import imgaug as ia
# img = imageio.imread("./test.jpg")

# ia.imshow(img)


# 添加Affine操作
from imgaug import augmenters as iaa

# ia.seed(4)
#
# # 实例化方法
# rotate = iaa.Affine(rotate=(-4, 45))
# img_aug = rotate(image=img)

# ia.imshow(img_aug)

# 这是对一张图片进行一种操作方式，但实际情况下，我们可能对一张图片做多种数据增强处理。这种情况下，我们就需要利用imgaug.augmenters.Sequential()来构造我们数据增强的pipline，该方法与torchvison.transforms.Compose()相类似。
# iaa.Sequential(children=None,   # Augmenter集合
#                random_order=False,  # 是否对每个batch使用不同顺序的Augmenter list
#                name=None,
#                deterministic=False,
#                random_state=None
#                )




# # 使用Sequential构建处理序列：
# # 构建处理序列
# aug_seq = iaa.Sequential([
#     iaa.Affine(rotate=(-25, 25)),
#     iaa.AdditiveGaussianNoise(scale=(10, 60)),
#     iaa.Crop(percent=(0, 0.2))
# ])
#
# # 对图片进行处理，image不可以省略，也不能写成images
# image_aug = aug_seq(image=img)
# # ia.imshow(image_aug)
#
#
# import numpy as np
# images = [img, img, img, img,]
# images_aug = aug_seq(images=images)
# ia.imshow(np.hstack(images_aug))
#
# # 使用Affine表示的是放射变换。
#
#
# # imgaug相较于其他的数据增强的库，有一个很有意思的特性，即就是我们可以通过imgaug.augmenters.Sometimes()对batch中的一部分图片应用一部分Augmenters,剩下的图片应用另外的Augmenters。
# # 示例代码如下：
# iaa.Sometimes(p=0.5,  # 代表划分比例
#               then_list=None,  # Augmenter集合。p概率的图片进行变换的Augmenters。
#               else_list=None,  #1-p概率的图片会被进行变换的Augmenters。注意变换的图片应用的Augmenter只能是then_list或者else_list中的一个。
#               name=None,
#               deterministic=False,
#               random_state=None)
#


# 对不同大小的图片进行处理





# imgaug在pytorch中的应用
import numpy as np
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import imgaug


tfs = transforms.Compose([
    iaa.Sequential([
        iaa.flip.Fliplr(p=0.5),
        iaa.flip.Flipud(p=0.5),
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.MultiplyBrightness(mul=(0.65, 1.35)),
    ]).augment_image,

    # 使用ToTensor()
    transforms.ToTensor()
])


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, n_images, n_classes, transform=None):
        # 图片的读取，建议使用imageio
        self.images = np.random.randint(0, 255, (n_images, 224, 224, 3),
                                        dtype=np.uint8)
        self.targets = np.random.randn(n_images, n_classes)
        self.transform = transform


    def __getitem__(self, item):
        image = self.images[item]
        target = self.targets[item]

        if self.transform:
            image = self.transform(image)
        return image, target


    def __len__(self):
        return len(self.images)


def worker_init_fn(worker_id):
    # 设计imgaug实例对象的随机种子
    imgaug.seed(np.random.get_state()[1][0] + worker_id)


mydata_ds = MyDataset(n_images=50, n_classes=10, transforms=tfs)
mydata_dl = DataLoader(mydata_ds, batch_size=64, num_workers=True,
                       worker_init_fn=worker_init_fn)


# 关于num_workers在Windows系统上只能设置成0，但是当我们使用Linux远程服务器时，可能使用不同的num_workers的数量，这是我们就需要注意worker_init_fn()函数的作用了。它保证了我们使用的数据增强在num_workers>0时是对数据的增强是随机的。

# 数据扩充是我们需要掌握的基本技能，除了imgaug以外，我们还可以去学习其他的数据增强库，包括但不局限于Albumentations，Augmentor。除去imgaug以外，我还强烈建议大家学下Albumentations，因为Albumentations跟imgaug都有着丰富的教程资源，大家可以有需求访问Albumentations教程。







# 6.6 使用argparse进行调参
# 使用argparse步骤：
# 1.创建ArgumentParser()对象
# 2.调用add_argument()方法添加参数
# 3.调用parse_args()解析参数




# 使用argparse示例代码如下：


import argparse

# 创建ArgumentParser()对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('-o', '--output', action='store_true',
    help="shows output")
# action = `store_true` 会将output参数记录为True
# type 规定了参数的格式
# default 规定了默认值
parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=1e-3')

parser.add_argument('--batch_size', type=int, required=True, help='input batch size')
# 使用parse_args()解析函数
args = parser.parse_args()

if args.output:
    print("This is some output")
    print(f"learning rate:{args.lr} ")





# 不写--时，argparse会对命令行参数按照参数位置进行严格解析

# # positional.py
# import argparse
#
# # 位置参数
# parser = argparse.ArgumentParser()
#
# parser.add_argument('name')
# parser.add_argument('age')
#
# args = parser.parse_args()
#
# print(f'{args.name} is {args.age} years old')
# 当我们不实用--后，将会严格按照参数位置进行解析。
#
# $ positional_arg.py Peter 23
# Peter is 23 years old


# 作者介绍自己使用argparse的方法：
import argparse


def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers, you had better put it '
                             '4 times of your gpu')

    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=64')

    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for, default=10')

    parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=1e-3')

    parser.add_argument('--seed', type=int, default=118, help="random seed")

    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to load a previous trained model if not empty (default empty)')
    parser.add_argument('--output', action='store_true', default=True, help="shows output")

    opt = parser.parse_args()

    if opt.output:
        print(f'num_workers: {opt.workers}')
        print(f'batch_size: {opt.batch_size}')
        print(f'epochs (niters) : {opt.niter}')
        print(f'learning rate : {opt.lr}')
        print(f'manual_seed: {opt.seed}')
        print(f'cuda enable: {opt.cuda}')
        print(f'checkpoint_path: {opt.checkpoint_path}')

    return opt


if __name__ == '__main__':
    opt = get_options()
# $ python
# config.py

num_workers: 0
batch_size: 4
epochs(niters): 10
learning
rate: 3e-05
manual_seed: 118
cuda
enable: True
checkpoint_path:
随后在train.py等其他文件，我们就可以使用下面的这样的结构来调用参数。

# 导入必要库
...
import config

opt = config.get_options()

manual_seed = opt.seed
num_workers = opt.workers
batch_size = opt.batch_size
lr = opt.lr
niters = opt.niters
checkpoint_path = opt.checkpoint_path


# 随机数的设置，保证复现结果
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


...

if __name__ == '__main__':
    set_seed(manual_seed)
    for epoch in range(niters):
        train(model, lr, batch_size, num_workers, checkpoint_path)
        val(model, lr, batch_size, num_workers, checkpoint_path)











