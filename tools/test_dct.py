import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F


# 量化系数
def ImplicitTrans(x, weight):
    table = torch.tensor([
        16, 16, 16, 16, 17, 18, 21, 24,
        16, 16, 16, 16, 17, 19, 22, 25,
        16, 16, 17, 18, 20, 22, 25, 29,
        16, 16, 18, 21, 24, 27, 31, 36,
        17, 17, 20, 24, 30, 35, 41, 47,
        18, 19, 22, 27, 35, 44, 54, 65,
        21, 22, 25, 31, 41, 54, 70, 88,
        24, 25, 29, 36, 47, 65, 88, 115]) / 255.0
    table = table.unsqueeze(-1) # 添加维度 table 的值是DCT的振幅A
    table = table.unsqueeze(-1)
    table = table.unsqueeze(-1)  # size=(64,1,1,1)

    temp = torch.empty(256, 1, 1, 1)
    factor = nn.Parameter(torch.ones_like(temp))
    bias = nn.Parameter(torch.zeros_like(temp))

    # table = table.cuda()
    # 设置kernel
    conv_shape = (256, 64, 1, 1)
    kernel = np.zeros(conv_shape, dtype='float32')
    r1 = math.sqrt(1.0 / 8)  # 公式的补偿系数 根号（1/8）
    r2 = math.sqrt(2.0 / 8)

    for i in np.arange(0.0, 8.0, 0.5):  # 要改成间隔是0.5步长，i --> 2i-1  16次
        _u = 2 * i + 1
        for j in np.arange(0.0, 8.0, 0.5):
            _v = 2 * j + 1
            index = i * 8 + j
            index1 = int(2*index - 1)
            for u in range(8):
                for v in range(8):
                    index2 = u * 8 + v
                    # 离散余弦逆变换
                    t = math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)
                    t = t * r1 if u == 0 else t * r2  # if u=0, t=t*r1
                    t = t * r1 if v == 0 else t * r2
                    kernel[index1, index2, 0, 0] = t  # size=(256,1,1,1)

    kernel = torch.from_numpy(kernel)  # [256, 64, 1, 1]
    new_table = torch.repeat_interleave(table, repeats=4, dim=0)  # 将table第0维度复制四个， [256, 1, 1, 1]  改64通道不行，idct的量化变构成了第二个维度的
    _table = new_table * factor + bias  # [256, 1, 1, 1]
    # print(_table)

    _kernel = kernel * _table  # size = torch.Size([256, 64, 1, 1])  # table不参与更新
    x = x * weight  # [16, 64, 16, 16]
    y = F.conv2d(input=x, weight=_kernel, stride=1)  # 将weight值传出来

    pixel_shuffle = nn.PixelShuffle(2)
    y = pixel_shuffle(y)
    return y, _table


def ImplicitTranCompensate(x):
    table = torch.tensor([
        16, 16, 16, 16, 17, 18, 21, 24,
        16, 16, 16, 16, 17, 19, 22, 25,
        16, 16, 17, 18, 20, 22, 25, 29,
        16, 16, 18, 21, 24, 27, 31, 36,
        17, 17, 20, 24, 30, 35, 41, 47,
        18, 19, 22, 27, 35, 44, 54, 65,
        21, 22, 25, 31, 41, 54, 70, 88,
        24, 25, 29, 36, 47, 65, 88, 115]) / 255.0
    table = table.unsqueeze(-1)  # 松开，添加维度
    table = table.unsqueeze(-1)
    table = table.unsqueeze(-1)  # size=(64,1,1,1)

    temp = torch.empty(256, 1, 1, 1)
    factor = nn.Parameter(torch.ones_like(temp))
    bias = nn.Parameter(torch.zeros_like(temp))
    conv_shape = (256, 64, 1, 1)
    kernel = np.zeros(conv_shape, dtype='float32')
    r1 = math.sqrt(1.0 / 8)  # 公式的补偿系数 根号（1/8）
    r2 = math.sqrt(2.0 / 8)

    for i in np.arange(0.0, 8.0, 0.5):  # 要改成间隔是0.5步长，i --> 2i-1  16次
        _u = 2 * i + 1
        for j in np.arange(0.0, 8.0, 0.5):
            _v = 2 * j + 1
            index = i * 8 + j
            index1 = int(2 * index - 1)
            for u in range(8):
                for v in range(8):
                    index2 = u * 8 + v
                    # 离散余弦逆变换求导，u是量化表，_u是时间轴
                    # t = math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)
                    t_pi = math.sin(_u*u*math.pi / 16) * (-_u*math.pi / 16) * math.cos(_v * v * math.pi / 16) + math.cos(_u*u*math.pi/16)*(-_v*math.pi / 16)*math.sin(_v * v * math.pi / 16)
                    t = t_pi * r1 if u == 0 else t_pi * r2  # if u=0, t=t*r1
                    t = t_pi * r1 if v == 0 else t_pi * r2
                    kernel[index1, index2, 0, 0] = t  # size=(256,1,1,1)

    kernel = torch.from_numpy(kernel)  # size=(256,1,1,1)
    new_table = torch.repeat_interleave(table, repeats=4, dim=0)  # 将table第0维度复制四个，
    _table = new_table * factor + bias
    _kernel = kernel * _table  # size = torch.Size([256, 64, 1, 1])

    weight = torch.ones(16, 64, 1, 1)
    x = x * weight
    y = F.conv2d(input=x, weight=_kernel, stride=1)
    pixel_shuffle = nn.PixelShuffle(2)
    y = pixel_shuffle(y)
    return y


conv_up = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
conv_ = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
conv_down = nn.Conv2d(in_channels=64 * 2 * 2, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
pixelUnshuffle = nn.PixelUnshuffle(downscale_factor=2)


if __name__ == "__main__":
    # compensate
    # input x = torch.ones(16, 64, 16, 16) [1, 64, 48, 48]
    x = torch.ones(16, 64, 16, 16)
    weight = torch.ones(16, 64, 1, 1)
    input, table = ImplicitTrans(x, weight)  # [16, 64, 32, 32])
    # print(table)
    temp = conv_up(x)
    tmp = conv_up(x)
    print('{}'.format(temp.equal(tmp)))
    print(temp)
    # print('input : {}'.format(input.shape))
    # print('x : {}'.format(x.shape))
    # print('kernel : {}'.format(kernel.shape))
    # print(kernel[20][1][0][0])
    # print(temp_kernel[20][1][0][0])
    # print(table[20][0][0][0])
    # if kernel[20][1][0][0] == temp_kernel[20][1][0][0]:
    # if kernel.equal(temp):
    #     print('Ture')
    # else:
    #     print('False')
    # print("{}".format(input_x.shape))
    #
    # input_x = conv_up(input_x)  # [16, 64, 16, 16]
    # print("{}".format(input_x.shape))
    #
    # input_x = ImplicitTranCompensate(input_x)  # [16, 64, 32, 32]
    # print("{}".format(input_x.shape))

    # input_x = pixelUnshuffle(input_x)  # [16, 256, 16, 16]
    # input_x = conv_down(input_x)  # [16, 64, 16, 16]
    # print("{}".format(input_x.shape))







