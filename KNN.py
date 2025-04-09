"""
@FileName：KNN.py
@Description：
@Author：liubeihefei
@Time：2025/4/5 11:56
"""

import torch
from Dataset import MakeDataset
import matplotlib.pyplot as plt
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 由于无训练参数，所以这里的KNN网络直接按类编写，不继承torch的类
class KNN:
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def preprocess(self, x):
        x = self.pool(x)
        x = self.pool(x)
        # 将图片展平成一维
        return torch.flatten(x)

    def test(self, k):
        # 存放分对-错的数量
        result = [0, 0]
        # 遍历每个测试样本
        cnt = 1

        #-------------------------------------------------------------------
        # 提前将整个训练集加载进内存
        train_data = []
        train_labels = []
        for i, data in enumerate(self.train_loader, 0):
            input, label = data
            input = self.preprocess(input)
            train_data.append(input)
            train_labels.append(label)
        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels)
        # # 放到显存，利用显卡计算
        train_data = train_data.to(device)
        train_labels = train_labels.to(device)
        #------------------------------------------------------------------

        for i_test, data_test in enumerate(self.test_loader, 0):
            # 打印测试信息
            print("第" + str(cnt) + "个测试样本")
            cnt += 1

            # 记录开始时间
            start_time = time.time()

            # 分离图片和标签
            input_test, label_test = data_test
            input_test = self.preprocess(input_test)
            # # 拿到显存
            input_test = input_test.to(device)
            label_test = label_test.to(device)

            # 存放前k个最相似的标签及对应的相似度
            # list = [(-1, 0)]
            list = []

            # 计算测试样本与整个训练集的相似度
            cs = torch.cosine_similarity(input_test, train_data, dim=1)
            # 找出tok-k个相似的相似度和对应索引
            values, indices = torch.topk(cs, k, 0)
            # 放入列表
            for i, j in zip(values, indices):
                list.append((train_labels[j], i))

            # # 遍历每个训练样本
            # for j_train, data_train in enumerate(self.train_loader, 0):
            #     input_train, label_train = data_train
            #     input_train = self.preprocess(input_train)
            #     # 计算当前样本与其的余弦相似度
            #     cs = torch.cosine_similarity(input_test, input_train, dim=0)
            #     # 将列表按相似度进行降序排序
            #     list = sorted(list, key=lambda x: x[1], reverse=True)
            #     # 判断当前样本是否可以入列表
            #     if cs > list[-1][1]:
            #         if len(list) < k:
            #             list.append((label_train, cs))
            #         else:
            #             list[-1] = ((label_train, cs))

            # 将列表中数量最多的类赋给当前测试样本，并与实际标签进行比较
            dict = {}
            for i in range(len(list)):
                key = int(list[i][0])
                dict[key] = dict.get(key, 0) + 1
            dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

            # 记录结束时间
            end_time = time.time()
            # 计算并打印执行时间
            print(f"执行时间：{end_time - start_time}秒")

            # 打印前k个标签与测试样本标签
            print("前k个的标签：", end="")
            for i in range(len(dict)):
                print(dict[i], end=" ")
            print("测试样本真实标签：" + str(int(label_test)))

            if dict[0][0] == int(label_test):
                result[0] += 1
                print("分类正确！")
            else:
                result[1] += 1
                print("分类错误！")
        # 最后返回在测试集上的准确率
        return result[0] / (result[0] + result[1])


if __name__ == '__main__':
    # 构造D0、D1、D2、D3数据集
    md = MakeDataset(1)
    md.make()
    # 选择训练集和测试集
    train_loader = md.get_trainloader('D0')
    test_loader = md.get_testloader()
    # test_loader = md.get_trainloader('D3')
    # 对三种k 1、3、5进行测试
    knn = KNN(train_loader, test_loader)
    k_list = [1, 3, 5]
    # 记录三种k对应的分类准确率
    result = []
    for k in k_list:
        res = knn.test(k)
        result.append(res)
    # 对结果进行可视化
    plt.figure(figsize=(6, 5))
    bars = plt.bar(k_list, result, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black', linewidth=0.7)
    # 设置y轴范围为0-1000
    plt.ylim(0, 1)
    # 添加横向网格线
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xlabel('k', labelpad=10)
    plt.ylabel('accuracy', labelpad=10)
    plt.xticks(k_list)
    # 在柱子上方显示数值（调整位置避免超出y轴）
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., min(height + 0.01, 1), "%.4f" % height, ha='center', va='bottom', fontsize=9)
    plt.savefig("./figure/knn_accurace_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化图片已保存为 '{'./figure/knn_accurace_comparison.png'}'")
    print(result)