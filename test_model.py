import math
import os
import time

import imageio
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from imageio import imread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Function

from torch.nn import Flatten
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import pickle

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# from no_act_new import net
from original import net

import os

now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

def save_image_c(data, n):
    output_dir = "./results/" + now + "/c"
    os.makedirs(output_dir, exist_ok=True)

    to_pil = ToPILImage()

    # 循环处理每个 batch 的 embedder_output
    for i in range(data.shape[0]):
        # 获取当前图片的张量
        image_tensor = data[i]  # 假设这里的张量是[3, 128, 128]
        image_example = to_pil(image_tensor)
        image_example.save(f"{output_dir}/c_{n}_{i}.png")

def save_image_wi(data, n):
    output_dir = "./results/" + now + "/wi"
    os.makedirs(output_dir, exist_ok=True)

    to_pil = ToPILImage()

    # 循环处理每个 batch 的 embedder_output
    for i in range(data.shape[0]):
        # 获取当前图片的张量
        image_tensor = data[i]  # 假设这里的张量是[1, 32, 32]
        image_example = to_pil(image_tensor)
        image_example.save(f"{output_dir}/wi_{n}_{i}.png")

def save_image_m(embedder_output, n):
    # 定义输出文件夹路径
    output_dir = "./results/" + now + "/m"
    os.makedirs(output_dir, exist_ok=True)

    to_pil = ToPILImage()

    # 循环处理每个 batch 的 embedder_output
    for i in range(embedder_output.shape[0]):
        # 获取当前图片的张量
        image_tensor = embedder_output[i]  # 假设这里的张量是[3, 128, 128]
        image_example = to_pil(image_tensor)
        image_example.save(f"{output_dir}/m_{n}_{i}.png")

        # # 转换张量为 numpy 数组，并调整维度顺序
        # image_numpy = image_tensor.cpu().numpy().transpose((1, 2, 0))
        #
        # # 创建 Matplotlib 图像对象
        # plt.imshow(image_numpy)
        #
        # # 将图片保存到文件
        # filename = os.path.join(output_dir, f"m_{n}_{i}.png")
        # plt.savefig(filename)
        #
        # # 关闭当前图像对象，释放资源
        # plt.close()
# matblib
def save_image_w(decoder_output, n):
    # 定义输出文件夹路径
    output_dir = "./results/" + now + "/w"
    os.makedirs(output_dir, exist_ok=True)
    to_pil = ToPILImage()
    # 循环处理每个 batch 的 embedder_output
    for i in range(decoder_output.shape[0]):
        # 获取当前图片的张量
        image_tensor = decoder_output[i]  # 假设这里的张量是[1, 32, 32]
        image_example = to_pil(image_tensor)
        image_example.save(f"{output_dir}/w_{n}_{i}.png")

        # # 转换张量为 numpy 数组，并调整维度顺序
        # image_numpy = image_tensor.cpu().numpy().transpose((1, 2, 0))
        #
        # # 创建 Matplotlib 图像对象
        # plt.imshow(image_numpy)
        #
        # # 将图片保存到文件
        # filename = os.path.join(output_dir, f"m_{n}_{i}.png")
        # plt.savefig(filename)
        #
        # # 关闭当前图像对象，释放资源
        # plt.close()

# 测试函数
def test(model, test_loader, hiding_data_loader):
    model.eval()
    x = 0
    for batch, (data, _) in enumerate(test_loader):
        print("batch:", batch)
        for ct, (hiding_data, _) in enumerate(hiding_data_loader):
            print("ct:", ct)
            data = data.to(device)
            hiding_data = hiding_data.to(device)

            save_image_wi(hiding_data, x)
            save_image_c(data, x)

            encoder_output = model.en.encoder(hiding_data)
            embedder_output = model.en.embedder(encoder_output, data, batch_size)  # 最后这个是batch_size
            print("embedder_output.shape:", embedder_output.shape)

            # 将embedder_output转换成图片，并保存
            save_image_m(embedder_output, x)
            invariance_output, weights = model.invariance(embedder_output)  # 这里的N是冗余，所以之后应该加上一个正则化项，提高这里的鲁棒性
            extractor_output = model.extractor(invariance_output)
            decoder_output = model.decoder(extractor_output, batch_size)  # 最后这个是batch_size
            print("decoder_output.shape:", decoder_output.shape)
            save_image_w(decoder_output, x)
            x += 1

def test_new(model, test_loader, hiding_data_loader):
    model.eval()
    x = 0
    length = len(test_loader)
    b = 0
    with torch.no_grad():
        for batch in range(length):
            print("batch:", batch)
            # 各取一个bs的数据进行组合
            data_iter = iter(test_loader)
            data, _ = next(data_iter)
            hiding_data_iter = iter(hiding_data_loader)
            hiding_data, _ = next(hiding_data_iter)
            data = data.to(device)
            hiding_data = hiding_data.to(device)

            save_image_wi(hiding_data, x)
            save_image_c(data, x)

            # 基本输出
            encoder_output = model.encoder(hiding_data)
            embedder_output = model.embedder(encoder_output, data, batch_size)  # 最后这个是batch_size
            save_image_m(embedder_output, x)

            invariance_output, weights = model.invariance(embedder_output)  # 这里的N是冗余，所以之后应该加上一个正则化项，提高这里的鲁棒性
            extractor_output = model.extractor(invariance_output)
            decoder_output = model.decoder(extractor_output, batch_size)  # 最后这个是batch_size
            save_image_w(decoder_output, x)
            b += calculate_BER_new(hiding_data, decoder_output)

            x += 1
    return b / length

def calculate_BER_new(hiding_data, decoder_output):
    # 将张量转换为 numpy 数组
    np_array = hiding_data.numpy().squeeze()
    # 将数值裁剪到 [0, 1] 范围
    np_array = np.clip(np_array, 0, 1)
    # 将数值转换为 0-255 的范围，使用round进行四舍五入
    np_array *= 255
    np_array = np.round(np_array).astype(np.uint8)
    # 将大于128的设置为255，小于等于128的设置为0
    np_array[np_array > 128] = 255
    np_array[np_array <= 128] = 0

    d = decoder_output.cpu().numpy().squeeze()
    d = np.clip(d, 0, 1)
    d *= 255
    d = np.round(d).astype(np.uint8)
    # 将d中大于128的设置为255，小于128的设置为0
    d[d > 128] = 255
    d[d <= 128] = 0


    # # 将图像转换为 numpy 数组
    # original_array = np.array(original_image)
    # received_array = np.array(received_image)

    # 计算不同像素的数量
    different_pixels = np.sum(np_array != d)

    # 计算误码率（BER）(不同的像素数 / 总像素数)
    total_pixels = np_array.size
    BER = different_pixels / total_pixels

    return BER

import skimage
def calculate_PSNR():
    path = 'E:\code\deepsc_hiding_torch\\results\\' + now + '\c\c_0_'
    path2 = 'E:\code\deepsc_hiding_torch\\results\\' + now + '\m\m_0_'
    path3 = 'E:\code\deepsc_hiding_torch\\results\\' + now + '\w\w_0_'
    path4 = 'E:\code\deepsc_hiding_torch\\results\\' + now + '\wi\wi_0_'

    total_psnr = 0
    total_ber = 0
    for i in range(100):
        img1 = imageio.v2.imread(path + str(i) + '.png')
        img2 = imageio.v2.imread(path2 + str(i) + '.png')
        img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
        psnr = PSNR(img1, img2)
        total_psnr += psnr
        print(psnr)
        image1 = Image.open(path3 + str(i) + '.png')
        image2 = Image.open(path4 + str(i) + '.png')
        ber = calculate_BER(image1, image2)
        total_ber += ber
        print(ber)
    print("total_psnr:", total_psnr / 100)
    print("total_ber:", total_ber / 100)


def calculate_BER(original_image, received_image):
    # 将图像转换为 numpy 数组
    original_array = np.array(original_image)
    received_array = np.array(received_image)

    # 计算不同像素的数量
    different_pixels = np.sum(original_array != received_array)

    # 计算误码率（BER）(不同的像素数 / 总像素数)
    total_pixels = original_array.size
    BER = different_pixels / total_pixels

    return BER

if __name__ == "__main__":
    # model = net(12).to(device)
    model = net(5).to(device)
    batch_size = 100

    checkpoint_path = 'E:\code\deepsc_hiding_torch\checkpoints\image_hiding\\2024-06-03-15_44_38\checkpoint_260_0.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    # 将输入数据移到 GPU 上
    data_dir = "E:\code\deepsc_hiding_torch\data\image\\test128"
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    hiding_data_dir = "E:\code\deepsc_hiding_torch\dataset\MNIST_data\min\\test\\test_new"
    hiding_data_transform = transforms.Compose([
        transforms.Grayscale(),  # 转换为灰度图像
        transforms.ToTensor()  # 转换为张量
    ])
    hiding_data_dataset = datasets.ImageFolder(root=hiding_data_dir, transform=hiding_data_transform)
    # hiding_data_dataset = CIFAR10(root='E:\code\deepsc_hiding_torch\dataset\dataset', train=False, download=True,
    #                               transform=hiding_data_transform)
    hiding_data_loader = DataLoader(hiding_data_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    b = test_new(model, test_loader, hiding_data_loader)
    calculate_PSNR()
    print("BER:", b)
    # image1 = Image.open('E:\code\deepsc_hiding_torch\\results\w\w_0_2.png')
    # image2 = Image.open('E:\code\deepsc_hiding_torch\\results\wi\wi_0_2.png')
    # calculate_BER(image1, image2)


