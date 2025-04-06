"""
@FileName：Dataset.py
@Description：
@Author：liubeihefei
@Time：2025/4/5 2:53
"""

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt


class MakeDataset:
    def __init__(self, batch_size):
        # 定义数据转换，图片归一化
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.batch_size = batch_size

        # 加载原始MNIST数据集
        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

        # 子训练集对应loader
        self.subloader = {}
        # 子训练集对应的各类的数量
        self.stats = {}

        # 定义采样比例和对应的子集名称
        self.sample_ratios = [0.1, 0.003, 0.001]  # 10%, 1%, 0.1%
        self.subset_names = ['D1', 'D2', 'D3']

    def make(self):
        # 获取原始训练集的标签
        train_labels = self.train_dataset.targets.numpy()
        for ratio, name in zip(self.sample_ratios, self.subset_names):
            # 为每个类别计算需要采样的数量
            samples_per_class = int(len(train_labels) / 10 * ratio)
            # 存储各个类子集的索引
            subset_indices = []
            # 对每个类别进行采样
            for class_id in range(10):
                # 获取当前类别的所有样本索引
                class_indices = np.where(train_labels == class_id)[0]

                # 随机采样，不重复
                sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                subset_indices.extend(sampled_indices)
            # 打乱顺序
            np.random.shuffle(subset_indices)
            # 创建Subset对象
            subset = Subset(self.train_dataset, subset_indices)
            # 创建DataLoader
            dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            # 存储到字典中
            self.subloader[name] = dataloader
            # 统计各类样本数量
            subset_labels = train_labels[subset_indices]
            label_counts = Counter(subset_labels)
            self.stats[name] = label_counts

    def visualize(self, save_path='./figure/mnist_subsets_distribution.png'):
        plt.figure(figsize=(15, 5))
        for idx, (name, counts) in enumerate(self.stats.items(), 1):
            plt.subplot(1, 3, idx)
            values = [counts[i] for i in range(10)]
            bars = plt.bar(range(10), values,
                           color=['skyblue', 'lightgreen', 'salmon'][idx - 1],
                           edgecolor='black', linewidth=0.7)
            # 设置y轴范围为0-1000
            plt.ylim(0, 1000)
            # 添加横向网格线
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.title(f'{name}\ntotal nums: {sum(values)}', pad=15)
            plt.xlabel('class', labelpad=10)
            plt.ylabel('nums', labelpad=10)
            plt.xticks(range(10))
            # 在柱子上方显示数值（调整位置避免超出y轴）
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.,
                         min(height + 20, 980),  # 确保标签不超过y轴上限
                         f'{int(height)}',
                         ha='center', va='bottom',
                         fontsize=9)
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化图片已保存为 '{save_path}'")

    def get_trainloader(self, name):
        if name == 'D0':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return self.subloader[name]

    def get_testloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    md = MakeDataset(1)
    md.make()
    md.visualize()
