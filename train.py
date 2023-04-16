import torch
import h5py
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from repvgg import create_RepVGG_A0
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model = create_RepVGG_A0(deploy=False).to(device)
train_model[0].load_state_dict(torch.load('model/RepVGG-A0-train.pth', map_location=device))

"""
启用batch normalization和drop out。
告诉BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN 层的参数是在不断变化的。
告诉 Dropout 层，你下面应该遮住一神经元
"""

# pytorch中的图像预处理包，用Compose把多个步骤整合到一起
transforms_my = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载数据集
dataset_train = ImageFolder(r'G:\emotion_detect-main\datasets\train', transform=transforms_my)
dataset_val = ImageFolder(r'G:\emotion_detect-main\datasets\val', transform=transforms_my)
batch_size = 32
dataloder_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dataloder_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)

# optimizer = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(train_model.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-8)
optimizer = torch.optim.Adagrad(train_model.parameters(), lr=0.0005, lr_decay=1e-4, weight_decay=1e-4)
""""
实现Adagrad优化方法(Adaptive Gradient)，Adagrad是一种自适应优化方法，是自适应的为各个参数分配不同的学习率。
这个学习率的变化，会受到梯度的大小和迭代次数的影响。梯度越大，学习率越小；梯度越小，学习率越大。
缺点是训练后期，学习率过小，因为Adagrad累加之前所有的梯度平方作为分母。
lr_decay (float, 可选) – 学习率衰减（默认: 0）
weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
"""

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
"""
optimer指的是网络的优化器
mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
factor 学习率每次降低多少，new_lr = old_lr * factor
patience=10，容忍网络的性能不提升的次数，高于这个次数就降低学习率
verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
min_lr,学习率的下限
eps ，适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
"""

loss_func = nn.CrossEntropyLoss()  # 交叉熵

train_loss, val_loss = [], []
train_acc, val_acc = [], []

epochs = 50
patience = 10
best_val_loss = float('inf')
num_bad_epochs = 0
min_acc = 0
best_epoch = 0

for epoch in range(epochs):
    train_loss_epoch, val_loss_epoch = 0, 0
    train_corrects, val_corrects = 0, 0
    # scheduler.step(val_loss_epoch)
    print('开始第{0}次迭代'.format(epoch + 1))

    train_model.train()  # 设置为训练模式
    """
    启用batch normalization和drop out。
    告诉BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN 层的参数是在不断变化的。
    告诉 Dropout 层，你下面应该遮住一神经元
    """
    for i, (data, label) in enumerate(dataloder_train):
        data, label = data.to(device), label.to(device)
        # data, label = data.cuda(), label.cuda()
        output = train_model(data)
        loss = loss_func(output, label)
        pre_label = torch.argmax(output, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item() * data.size(0)
        train_corrects += torch.sum(pre_label == label.data)
        # print('already train{0} / {1} examples'.format(i + 1, len(dataloder_train)))

    train_loss_epoch /= len(dataset_train.targets)
    train_acc_epoch = train_corrects / len(dataset_train.targets)
    print('第{0}epoch的train_loss为{1}'.format(epoch + 1, train_loss_epoch))
    print('第{0}epoch的train_acc为{1}'.format(epoch + 1, train_acc_epoch))
    train_loss.append(train_loss_epoch)
    train_acc.append(train_acc_epoch)

    train_model.eval()
    for j, (data_val, label_val) in enumerate(dataloder_val):
        data_val, label_val = data_val.to(device), label_val.to(device)
        # data_val, label_val = data_val.cuda(), label_val.cuda()
        output = train_model(data_val)
        loss = loss_func(output, label_val)
        pre_label = torch.argmax(output, 1)
        val_loss_epoch += loss.item() * data_val.size(0)
        val_corrects += torch.sum(pre_label == label_val.data)
        # print('already val{0} / {1} examples'.format(j + 1, len(dataloder_val)))

    val_loss_epoch /= len(dataset_val.targets)
    val_acc_epoch = val_corrects / len(dataset_val.targets)
    print('第{0}epoch的val_loss为{1}'.format(epoch + 1, val_loss_epoch))
    print('第{0}epoch的val_acc为{1}'.format(epoch + 1, val_acc_epoch))
    val_loss.append(val_loss_epoch)
    val_acc.append(val_acc_epoch)

    # 每隔10轮保存模型权重
    if (epoch + 1) % 10 == 0:
        torch.save(train_model.state_dict(), './utils/train_model_{0}.pth'.format(epoch+1))
    # 保存最好的模型权重
    if val_acc_epoch > min_acc:
        min_acc = val_acc_epoch
        best_epoch = epoch
        print(f"save best model, 第{epoch + 1}轮")
        torch.save(train_model.state_dict(), 'model/best_model.pth')

    if epoch - best_epoch >= 5:
        print(f"连续5轮准确率没有提升，停止训练。")
        break

# matplot_loss(train_loss, val_loss)

file = h5py.File('./utils/info.h5', 'w')
file['train_loss'] = torch.tensor(train_loss)
file['train_acc'] = torch.tensor(train_acc)
file['val_loss'] = torch.tensor(val_loss)
file['val_acc'] = torch.tensor(val_acc)
file.close()

# 计算混淆矩阵
train_model.eval()
val_preds, val_labels = [], []

for data_val, label_val in dataloder_val:
    data_val, label_val = data_val.to(device), label_val.to(device)
    output = train_model(data_val)
    pre_label = torch.argmax(output, 1)
    val_preds.extend(pre_label.detach().cpu().numpy())
    val_labels.extend(label_val.detach().cpu().numpy())

cm = confusion_matrix(val_labels, val_preds)
print("混淆矩阵：")
print(cm)

# 绘制混淆矩阵
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel("预测值")
plt.ylabel("真实值")
plt.title("混淆矩阵")
plt.show()
