"""
@FileName：Augmentation.py
@Description：
@Author：liubeihefei
@Time：2025/4/6 14:09
"""

import numpy as np
import cv2 as cv
import torchvision.transforms

from Dataset import MakeDataset
import torch
from torchvision import transforms


class Aug:
    def __init__(self):
        self.mean = 0.1307
        self.std = 0.3081

    # 将tensor转化为图片
    def tensor_numpy(self, image):
        clean = image.clone().detach().cpu().squeeze(0)  # 去掉batch通道 (batch, C, H, W) --> (C, H, W)
        clean[0] = clean[0] * self.std + self.mean  # 数据去归一化
        clean = np.around(clean.mul(255))  # 转换到颜色255 [0, 1] --> [0, 255]
        clean = np.uint8(clean).transpose(1, 2, 0)  # 换三通道 (C, H, W) --> (H, W, C)
        return clean

    # 展示图片
    def show(self, x):
        cv.imshow("figure", x)
        cv.waitKey(0)

    # 综合数据增强，分三个等级，low、mid、high，分别将数据变为2、8、16倍
    def work(self, x, instruction):
        # 拿到图像左上角像素点的值（用来后续填充）
        fill_temp = float(x[0][0][0][0])
        # 若为low，则只加个高斯模糊返回即可
        if instruction == 'low':
            transform = torchvision.transforms.GaussianBlur(3)
            temp = transform(x)
            return torch.cat((x, temp), dim=0)
        # 若为mid，则返回4张旋转随机角度，4张平移随机距离的tensor（不叠加）
        if instruction == 'mid':
            transform_1 = torchvision.transforms.RandomAffine(20, fill=fill_temp)
            transform_2 = torchvision.transforms.RandomAffine(0, (0.1, 0.1), fill=fill_temp)
            temp = x
            for i in range(4):
                x1 = transform_1(x)
                x2 = transform_2(x)
                temp = torch.cat((temp, x1), dim=0)
                if i != 3:
                    temp = torch.cat((temp, x2), dim=0)
            return temp
        # 若为high，则返回18张高斯+旋转+平移+缩放
        if instruction == 'high':
            transform_1 = torchvision.transforms.GaussianBlur(3)
            transform_2 = torchvision.transforms.RandomAffine(20, (0.1, 0.1), (0.8, 1), fill=fill_temp)
            temp = x
            for i in range(15):
                x1 = transform_1(x)
                x2 = transform_2(x1)
                temp = torch.cat((temp, x2), dim=0)
            return temp

    # 保存图片
    def save(self, x):
        imgs = []
        for img in x:
            img = img.unsqueeze(0)
            img = aug.tensor_numpy(img)
            imgs.append(img)
        # 拼接图片
        cnt = len(imgs)
        if cnt == 2:
            result = np.concatenate([imgs[0], imgs[1]], axis=1)
        elif cnt == 8:
            temp1 = imgs[0]
            temp2 = imgs[4]
            for i in range(3):
                temp1 = np.concatenate([temp1, imgs[i + 1]], axis=1)
                temp2 = np.concatenate([temp2, imgs[i + 5]], axis=1)
            result = np.vstack((temp1, temp2))
        elif cnt == 16:
            final_img = []
            # 分4*4进行拼接
            for i in range(4):
                temp = imgs[i * 4]
                for j in range(3):
                    temp = np.concatenate([temp, imgs[i * 4 + j + 1]], axis=1)
                final_img.append(temp)
            result = final_img[0]
            for i in range(3):
                result = np.vstack((result, final_img[i + 1]))
        cv.imwrite('./figure/aug.png', result)
        print(f"可视化图片已保存为 '{'./figure/aug.png'}'")


if __name__ == '__main__':
    md = MakeDataset(1)
    train_loader = md.get_trainloader('D0')

    aug = Aug()

    flag = True
    for data in train_loader:
        if flag:
            inputs, target = data
            inputs = aug.work(inputs, 'high')
            aug.save(inputs)
            # for ip in inputs:
            #     ip = ip.unsqueeze(0)
            #     ip = aug.tensor_numpy(ip)
            #     aug.show(ip)
            flag = False
        else:
            break
