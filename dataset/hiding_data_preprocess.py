"""
hiding_data用的是CIFAR10数据集
将数据集中的图片转换为32*32*1的图片
"""
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
#
# # tf是将图像转换成32*32*1的图片
# tf = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((32, 32)),
#     transforms.ToTensor()
# ])
#
# # 将数据集保存到当前目录下的dataset文件夹中
# train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=tf, download=True)
#
# # 展示第一张图片
# print(train_set[0][0].shape)


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Combofish
# Filename: main.py

# from icecream import ic
# from torchvision import datasets
# from tqdm import tqdm
# import os
#
#
# train_data = datasets.MNIST(root="./MNIST_data/", train=True, download=True)
# test_data = datasets.MNIST(root="./MNIST_data/", train=False, download=True)
# saveDirTrain = './DataImages-Train'
# saveDirTest = './DataImages-Test'
#
# if not os.path.exists(saveDirTrain):
#     os.mkdir(saveDirTrain)
# if not os.path.exists(saveDirTest):
#     os.mkdir(saveDirTest)
#
# ic(len(train_data), len(test_data))
# ic(train_data[0])
# ic(train_data[0][0])
#
#
# def save_img(data, save_path):
#     for i in tqdm(range(len(data))):
#         img, label = data[i]
#         img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))
#
#
# save_img(train_data, saveDirTrain)
# save_img(test_data, saveDirTest)


from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(32, 32), grayscale=False):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)

        # 调整图像大小
        img_resized = img.resize(target_size)

        # 转换为灰度图像
        if grayscale:
            img_resized = img_resized.convert('L')

        # 保存调整大小后的图像
        output_path = os.path.join(output_folder, image_file)
        img_resized.save(output_path)

# 指定输入和输出文件夹路径
input_folder = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\dataset\DataImages-Train"
output_folder = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\dataset\MNIST_data\MNIST_32_train"

# 调整图像大小并保存到输出文件夹中
resize_images(input_folder, output_folder, grayscale=True)

