


# pytorchvision是torch官方提供的图像处理工具包；


# 定义读取FashionMinist数据集的代码：
class FMDataset(Dataset):
    def __init__(self, df, transform = None):
        self.transform = transform
        # 下面代码中，表示从batch中每张图像中除掉第一个index = 0位置的label标签
        self.images = df.iloc[:, 1:].values.astype(np.unit8)
        self.label = df.iloc[:, 0].values


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


# 在构建训练和测试数据集完成后，需要定义DataLoader类，以便在训练和测试时加载数据：

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# 读入后，我们可以做一些数据可视化操作，主要是验证我们读入的数据是是否正确
import matplotlib.pyplot as plt
image, label = next(iter(train_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")
torch.Size([256, 1, 28, 28])
torch.Size([256])
<matplotlib.image.AxesImage at 0x7f19a043cc10>


# 模型设计
# 由于任务较为简单，这里我们手搭一个CNN，而不考虑当下各种模型的复杂结构，模型构建完成后，将模型放到GPU上用于训练。
# 代码如下：
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

model = Net()
model = model.cuda()
# model = nn.DataParallel(model).cuda()   # 多卡训练时的写法，之后的课程中会进一步讲解



# 设定损失函数
# 使用torch.nn模块自带的CrossEntropy损失
# PyTorch会自动把整数型的label转为one-hot型，用于计算CE loss
# 这里需要确保label是从0开始的，同时模型不加softmax层（使用logits计算）,这也说明了PyTorch训练中各个部分不是独立的，需要通盘考虑
#
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=[1,1,1,1,3,1,1,1,1,1])


# 使用Adam优化器，代码如下：
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 训练代码如下：
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))



# 验证代码：
def val(epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))



# 开始训练并在每个epoch中进行验证，打印日志：
for epoch in range(1, epochs+1):
    train(epoch)
    val(epoch)



# 模型保存
# 训练完成后，可以使用torch.save保存模型参数或者整个模型，也可以在训练过程中保存模型
# 这部分会在后面的课程中详细介绍
#
# save_path = "./FahionModel.pkl"
# torch.save(model, save_path)


# 总结上述过程：
# 1.首先定义Dataset类；
# 2.定义DataLoader类；
# 3.设置训练需要的参数：epoch最大值，batch_size大小，初始化参数方法选择，初始化学习率，
# 设置加载数据集是否使用多线程，设置GPU index等；
# 4.加载数据之后，根据模型需要的大小尺寸进行调整和transform变换；
# 5.定义模型；
# 6.选择Loss Function；
# 7.设置优化器；
# 8.设置训练/验证的模块；
# 9.保存模型；




