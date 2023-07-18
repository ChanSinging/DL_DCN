from torch.autograd import Variable
from torch.nn import init, Conv2d, Sequential
from torch import nn as nn
from torch.nn import functional as F

import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms

from PIL import Image
from scipy.ndimage.filters import gaussian_filter


random.seed(10)
np.random.seed(10)

PIL_TRANS = torchvision.transforms.ToPILImage()
TENSOR_TRANS = torchvision.transforms.ToTensor()
class feLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(feLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Add_Gates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 3, padding=3 // 2)
        self.Dot_Gates = nn.Conv2d(input_size*2, input_size, 3, 1, 1)
        self.Out_Gates = nn.Conv2d(input_size*2, input_size, 3, 1, 1)
        # self.enhence = nn.Conv2d(input_size, input_size, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, input_, prev_hidden, prev_cell):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        # generate empty prev_state, if None is provided
        if prev_hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))

        # print(prev_hidden.size())

        prev_hidden = prev_hidden.cuda().detach()
        prev_cell = prev_cell.cuda().detach()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        Add_gates = self.Add_Gates(stacked_inputs)
        remember_gate = torch.sigmoid(self.Dot_Gates(stacked_inputs))
        out = self.lrelu(self.Out_Gates(stacked_inputs))

        # chunk across channel dimension
        in_gate, cell_gate = Add_gates.chunk(2, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = prev_cell + (in_gate * cell_gate)
        # compute current cell and hidden state
        hidden = out * torch.tanh(cell)
        cell = remember_gate * cell
        # cell = cell + self.lrelu(self.enhence(prev_cell))
        return hidden, hidden, cell

def load_image(file):
    """
    读取给定的图像文件
    :return:
    """
    image = Image.open(file)
    image = image.convert('RGB')

    return image


def get_data_set_list(folder_list_path, shuffle=False):
    """
    读取数据集文件名。
    :param folder_list_path:
    :param shuffle:
    :return:
    """
    folder_list = open(folder_list_path, 'rt').read().splitlines()
    if shuffle:
        random.shuffle(folder_list)

    files_list = []
    for folder in folder_list:
        pattern = os.path.join(folder, 'truth/*.png')
        files = sorted(glob.glob(pattern))
        files_list.append(files)

    return files_list


def get_gaussian_filter(size, sigma):
    """
    生成高斯过滤器。
    :param size: 过滤器大小
    :param sigma: 标准差
    :return:
    """
    template = np.zeros((size, size))
    template[size//2, size//2] = 1
    return gaussian_filter(template, sigma)


def down_sample_with_blur(images, kernel, scale):
    """
    对图片进行高斯模糊和下采样。
    :param images:
    :param kernel:
    :param scale:
    :return:
    """
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    pad_height = kernel_height - 1
    pad_width = kernel_width - 1

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = (pad_left, pad_right, pad_top, pad_bottom)

    kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(3, 1, 1, 1)

    reflect_layer = torch.nn.ReflectionPad2d(pad_array)

    padding_images = reflect_layer(images)
    # 在 pytorch 中通过设置 groups 参数来实现 depthwise_conv。
    # 详情参考：https://www.jianshu.com/p/20ba3d8f283c
    output = torch.nn.functional.conv2d(padding_images, kernel, stride=scale, groups=3)

    return output

class NonLocalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, sub_sample=1, nltype=0):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub_sample = sub_sample
        self.nltype = nltype

        self.convolution_g = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.pooling_g = torch.nn.AvgPool2d(self.sub_sample, self.sub_sample)

        self.convolution_phi = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.pooling_phi = torch.nn.AvgPool2d(self.sub_sample, self.sub_sample)

        self.convolution_theta = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.convolution_y = torch.nn.Conv2d(self.out_channels, self.in_channels, 1, 1, 0)

        self.LSTM = feLSTM()
        self.fusion_LS = nn.Conv2d(2 * 64, 64, 1, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, input_x, last_h, last_c):
        batch_size, in_channels, height, width = input_x.shape

        assert self.nltype <= 2, ValueError("nltype must <= 2")
        # g
        g = self.convolution_g(input_x)
        if self.sub_sample > 1:
            g = self.pooling_g(g)
        # phi
        if self.nltype == 0 or self.nltype == 2:
            phi = self.convolution_phi(input_x)
        elif self.nltype == 1:
            phi = input_x
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))
        if self.sub_sample > 1:
            phi = self.pooling_phi(phi)

        # theta
        if self.nltype == 0 or self.nltype == 2:
            theta = self.convolution_theta(input_x)
        elif self.nltype == 1:
            theta = input_x
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))

        g_x = g.reshape([batch_size, -1, self.out_channels])
        theta_x = theta.reshape([batch_size, -1, self.out_channels])
        phi_x = phi.reshape([batch_size, -1, self.out_channels])
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)

        f_b, f_c, f_s = f.size()
        # b,63, h*w//64, h*w
        LSTM_in = f.view(f_b, 64, f_c // 8, f_s//8)
        LSTM_out, next_h, next_c = self.LSTM(LSTM_in, last_h, last_c)
        LSTM_in = self.fusion_LS(torch.cat((LSTM_in, LSTM_out), dim=1))
        f = LSTM_in.view(f_b, f_c, f_s)

        if self.nltype <= 1:
            f = torch.exp(f)
            f_softmax = f / f.sum(dim=-1, keepdim=True)
        elif self.nltype == 2:
            self.relu(f)
            f_mean = f.sum(dim=2, keepdim=True)
            f_softmax = f / f_mean
        else:
            raise ValueError('nltype can not be: {}'.format(self.nltype))

        y = torch.matmul(f_softmax, g_x)
        y = y.reshape([batch_size, self.out_channels, height, width])
        z = self.convolution_y(y)

        return z, next_h, next_c

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)  # (n, bs, bs, c//bs^2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (n, c//bs^2, h, bs, w, bs)
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)  # (n, c//bs^2, h * bs, w * bs)
        return x


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)  # (n, c, h//bs, bs, w//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (n, bs, bs, c, h//bs, w//bs)
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)  # (n, c*bs^2, h//bs, w//bs)
        return x


class PFNL(nn.Module):
    def __init__(self):
        """
        模型参数、神经网路层初始化
        :param args: 每个样本包含的帧数
        """
        super(PFNL, self).__init__()
        self.n_frames = 2
        # self.train_size = (args.in_size, args.in_size)
        # self.eval_size = args.eval_in_size

        # 在输入为正方矩阵的简单情况下，padding = (k_size-1)/2 时为 SAME
        self.convolution_layer0 = Sequential(Conv2d(3, 64, 5, 1, 2), torch.nn.LeakyReLU())
        init.xavier_uniform_(self.convolution_layer0[0].weight)

        self.convolution_layer1 = nn.ModuleList([
            Sequential(
                Conv2d(64, 64, 3, 1, (3-1) // 2),
                torch.nn.LeakyReLU()
            )
            for _ in range(20)])

        self.convolution_layer10 = nn.ModuleList([
            Sequential(
                Conv2d(self.n_frames * 64, 64, 1, 1, 0),
                torch.nn.LeakyReLU()
            )
            for _ in range(20)])

        self.convolution_layer2 = nn.ModuleList([
            Sequential(
                Conv2d(2 * 64, 64, 3, 1, (3-1) // 2),
                torch.nn.LeakyReLU()
            )
            for _ in range(20)])

        # xavier初始化参数
        for i in range(20):
            init.xavier_uniform_(self.convolution_layer1[i][0].weight)
            init.xavier_uniform_(self.convolution_layer10[i][0].weight)
            init.xavier_uniform_(self.convolution_layer2[i][0].weight)

        self.convolution_merge_layer1 = Sequential(
            Conv2d(self.n_frames * 64, 48, 3, 1, 1), torch.nn.LeakyReLU()
        )
        init.xavier_uniform_(self.convolution_merge_layer1[0].weight)

        self.convolution_merge_layer2 = Sequential(
            Conv2d(48 // (2 * 2), 12, 3, 1, 1), torch.nn.LeakyReLU()
        )
        init.xavier_uniform_(self.convolution_merge_layer2[0].weight)

        # 参考：https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        self.space_to_depth, self.depth_to_space = SpaceToDepth(2), DepthToSpace(2)
        self.nonlocal_block = NonLocalBlock(2*2*3*self.n_frames, 3*self.n_frames*4, 1, 1)

    def forward(self, input_image, last_h, last_c):
        # 注意！输入图片的 shape 应该变为 batch_size * n_frames * channel * width * height
        input0 = [input_image[:, i, :, :, :] for i in range(self.n_frames)]
        input0 = torch.cat(input0, 1)
        # b,t,c,h,w --> b, t*c, h, w

        # print("input0:{}".format(input0.size()))
        input1 = self.space_to_depth(input0)
        # print("input1:{}".format(input1.size()))
        if last_h is None:
            input1, next_h, next_c = self.nonlocal_block(input1, None, None)
        else:
            input1, next_h, next_c = self.nonlocal_block(input1, last_h, last_c)
        input1 = self.depth_to_space(input1)
        input0 += input1

        input0 = torch.split(input0, 3, dim=1)
        input0 = [self.convolution_layer0(frame) for frame in input0]

        # basic = input_image[:, 0, :, :, :].squeeze(0)
        basic = F.interpolate(
            input_image[:, 0, :, :, :], scale_factor=4, mode='bilinear', align_corners=False)

        for i in range(20):
            input1 = [self.convolution_layer1[i](frame) for frame in input0]
            base = torch.cat(input1, 1)
            base = self.convolution_layer10[i](base)

            input2 = [torch.cat([base, frame], 1) for frame in input1]
            input2 = [self.convolution_layer2[i](frame) for frame in input2]
            input0 = [torch.add(input0[j], input2[j]) for j in range(self.n_frames)]

        merge = torch.cat(input0, 1)
        merge = self.convolution_merge_layer1(merge)

        large = self.depth_to_space(merge)
        output = self.convolution_merge_layer2(large)
        output = self.depth_to_space(output)

        return output+basic, next_h, next_c

    @staticmethod
    def perform_bicubic(image_tensor, scale):
        """
        对 tensor 类型的图像进行双三次线性差值。
        :param image_tensor:
        :param scale: 放大倍数
        :return:
        """
        n, c, h, w = image_tensor.shape

        image = PIL_TRANS(image_tensor)  # 注意在图片中会变成 w * h
        image = image.resize((w * scale, h * scale), Image.BICUBIC)
        output = TENSOR_TRANS(image)

        return output