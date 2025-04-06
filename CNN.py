"""
@FileName：CNN.py
@Description：
@Author：liubeihefei
@Time：2025/4/6 8:55
"""

import torch
import torch.nn.functional as F
from Dataset import MakeDataset
import matplotlib.pyplot as plt
import os
from Augmentation import Aug

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义两层卷积
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        # 定义池化层
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        # 定义线形层
        self.fc1 = torch.nn.Linear(500, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))  # 图片此时为10 * 12 * 12
        x = F.relu(self.pooling(self.conv2(x)))  # 图片此时为20 * 5 * 5
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))  # 全连接500->100
        x = self.fc2(x)          # 全连接100->10
        return x


def reinit_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def train(epoch, model, train_loader, criterion, optimizer, aug=None, instruction='high'):
    # 记录每代训练损失
    temp_loss = 0.0
    running_loss = []
    # 记录每代的准确率
    total = 0
    correct = 0
    running_acc = []
    for i in range(epoch):
        # 打印测试信息
        print("第" + str(i + 1) + "次训练")
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data

            # 如果有数据增强器
            if aug is not None:
                # 对图像进行增强
                temp = inputs[0]
                temp = temp.unsqueeze(0)
                for ip in inputs:
                    ip = ip.unsqueeze(0)
                    ip = aug.work(ip, instruction)
                    temp = torch.cat((temp, ip), dim=0)
                inputs = temp[1:]
                # 对相应标签进行复制
                dict = {'low': 2, 'mid': 8, 'high': 16}
                cnt = dict.get(instruction)
                temp = torch.empty(0, 1)
                for t in target:
                    t = t.repeat(cnt, 1)
                    temp = torch.cat((temp, t), dim=0)
                target = temp
                target = target.squeeze(1).long()

            inputs, target = inputs.to(device), target.to(device)
            # 先对梯度清零
            optimizer.zero_grad()
            # forward + backward + update
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
        # 记录每代的信息
        running_acc.append(correct / total)
        running_loss.append(temp_loss)
        correct = 0
        total = 0
        temp_loss = 0.0

    # 可视化训练过程
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(100), running_loss, label='train loss', linewidth=1, color='skyblue', marker='o', markerfacecolor='black', markersize=1.5)
    plt.xlabel('epoch')
    plt.ylabel('train loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(100), running_acc, label='train acc', linewidth=1, color='skyblue', marker='o', markerfacecolor='black', markersize=1.5)
    # 添加横向网格线
    plt.xlabel('epoch')
    plt.ylabel('train acc')
    plt.savefig('./figure/train.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化图片已保存为 '{'./figure/train.png'}'")


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


if __name__ == '__main__':
    # 准备模型和数据集
    model = CNN().to(device)
    md = MakeDataset(4)
    md.make()
    train_loader = md.get_trainloader('D3')
    test_loader = md.get_testloader()
    # 准备训练损失计算器和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # 数据增强器
    aug = None
    #----------------------单数据集训练------------------------
    # # 训练
    # train(100, model, train_loader, criterion, optimizer)
    # # 测试
    # acc = test(model, test_loader)
    # print("测试准确率为：" + str("%.2f" % (acc * 100)) + "%")
    #----------------------多数据集训练------------------------
    dataset = ['D3', 'D2', 'D1']
    test_acc = []
    for d in dataset:
        # 重新初始化模型
        reinit_model(model)
        train_loader = md.get_trainloader(d)
        train(100, model, train_loader, criterion, optimizer)
        acc = test(model, test_loader)
        test_acc.append(acc)
    plt.figure(figsize=(5, 5))
    bars = plt.bar(dataset, test_acc, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', linewidth=0.7)
    # 在柱子上方显示数值（调整位置避免超出y轴）
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., min(height + 0.005, 1.0), "%.2f" % height, ha='center', va='bottom', fontsize=9)
    plt.xlabel('dataset')
    plt.ylabel('acc')
    plt.title("test acc based on different training dataset")
    plt.savefig('./figure/test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化图片已保存为 '{'./figure/test.png'}'")
    #----------------------数据增强对比训练------------------------
    # # 无数据增强
    # reinit_model(model)
    # train(100, model, train_loader, criterion, optimizer, aug)
    # acc1 = test(model, test_loader)
    # # 有数据增强+low
    # aug = Aug()
    # reinit_model(model)
    # train(100, model, train_loader, criterion, optimizer, aug, instruction='low')
    # acc2 = test(model, test_loader)
    # # 有数据增强+low
    # reinit_model(model)
    # train(100, model, train_loader, criterion, optimizer, aug, instruction='mid')
    # acc3 = test(model, test_loader)
    # # 有数据增强+low
    # reinit_model(model)
    # train(100, model, train_loader, criterion, optimizer, aug, instruction='high')
    # acc4 = test(model, test_loader)
    #
    # plt.figure(figsize=(5, 5))
    # bars = plt.bar(['D3_none', 'D3_low', 'D3_mid', 'D3_high'], [acc1, acc2, acc3, acc4],
    #                color=['gray', 'skyblue', 'lightgreen', 'salmon'], edgecolor='black', linewidth=0.7)
    # # 在柱子上方显示数值（调整位置避免超出y轴）
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., min(height + 0.005, 1.0), "%.2f" % height, ha='center', va='bottom', fontsize=9)
    # plt.xlabel('different aug')
    # plt.ylabel('acc')
    # plt.title("test acc based on different aug")
    # plt.savefig('./figure/aug_accurace_comparison.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"可视化图片已保存为 '{'./figure/aug_accurace_comparison.png'}'")


