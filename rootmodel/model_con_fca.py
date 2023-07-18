import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *

from modules.modulated_deform_conv import _ModulatedDeformConv
from modules.modulated_deform_conv import ModulatedDeformConvPack
from Fcanet import MultiSpectralAttentionLayer


class maxpooldown(nn.Module):
    def __init__(self, scale=2, n_channel=4):
        super(maxpooldown, self).__init__()
        self.convD = nn.Conv2d(n_channel, n_channel, 7, padding=7 // 2, stride=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv = nn.Conv2d(n_channel, n_channel, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # input: b * 64 * 64 *64
        # return: b * 64 * 16 * 16
        out = self.convD(x)
        out = self.pool(out)
        out = self.lrelu(self.conv(out))
        return out


class GetWeight(nn.Module):  # 估计自适应量化表
    def __init__(self, channel=64):
        super(GetWeight, self).__init__()
        self.downsample = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 8, channel * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 4, channel * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, size, _ = x.size()  # torch.Size([1, 32, 24, 24])
        c = c
        x = self.downsample(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  # [16, 32, 1, 1]


class ImplicitTrans(nn.Module):
    def __init__(self, in_channels):
        super(ImplicitTrans, self).__init__()
        self.table = torch.tensor([
            16, 16, 16, 16, 17, 18, 21, 24,
            16, 16, 16, 16, 17, 19, 22, 25,
            16, 16, 17, 18, 20, 22, 25, 29,
            16, 16, 18, 21, 24, 27, 31, 36,
            17, 17, 20, 24, 30, 35, 41, 47,
            18, 19, 22, 27, 35, 44, 54, 65,
            21, 22, 25, 31, 41, 54, 70, 88,
            24, 25, 29, 36, 47, 65, 88, 115]) / 255.0  # .reshape(8, 8)
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)

        temp = torch.empty(256, 1, 1, 1)
        self.factor = nn.Parameter(torch.ones_like(temp))
        self.bias = nn.Parameter(torch.zeros_like(temp))
        self.table = self.table.cuda()

        conv_shape = (256, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')
        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in np.arange(0.0, 8.0, 0.5):  # 0.5倍的采样率
            _u = 2 * i + 1  # i 是 8*8像素块的横坐标和纵坐标
            for j in np.arange(0.0, 8.0, 0.5):
                _v = 2 * j + 1  # 2*j+1
                index = i * 8 + j
                index1 = int(2 * index - 1)
                for u in range(8):  # u, v is dct系数表
                    for v in range(8):
                        index2 = u * 8 + v
                        # 离散余弦逆变换
                        t = math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)
                        t = t * r1 if u == 0 else t * r2  # if u=0, t=t*r1
                        t = t * r1 if v == 0 else t * r2
                        kernel[index1, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, x, weight):  # -0.5<x<0.5 weight
        new_table = torch.repeat_interleave(self.table, repeats=4, dim=0)  # 将table第0维度复制四个，
        _table = new_table * self.factor + self.bias  # table is 量化系数, factor and bias are learnable
        _kernel = self.kernel * _table
        x = x * weight  # tensor x is [4,64,16,16]  x transform to compensate

        y = F.conv2d(input=x, weight=_kernel, stride=1)  # y size:torch.Size([1, 256, 24, 24])

        pixel_shuffle = nn.PixelShuffle(2)
        y = pixel_shuffle(y)  # [1,64,48,48]

        return y, _table, x


class ImplicitTransCompensate(nn.Module):
    def __init__(self, in_channels):
        super(ImplicitTransCompensate, self).__init__()
        self.table = torch.tensor([
            16, 16, 16, 16, 17, 18, 21, 24,
            16, 16, 16, 16, 17, 19, 22, 25,
            16, 16, 17, 18, 20, 22, 25, 29,
            16, 16, 18, 21, 24, 27, 31, 36,
            17, 17, 20, 24, 30, 35, 41, 47,
            18, 19, 22, 27, 35, 44, 54, 65,
            21, 22, 25, 31, 41, 54, 70, 88,
            24, 25, 29, 36, 47, 65, 88, 115]) / 255.0  # .reshape(8, 8)
        self.table = self.table.unsqueeze(-1)  # 设置维数
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)

        temp = torch.empty(256, 1, 1, 1)
        self.factor = nn.Parameter(torch.ones_like(temp))
        self.bias = nn.Parameter(torch.zeros_like(temp))
        self.table = self.table.cuda()

        conv_shape = (256, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')
        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in np.arange(0.0, 8.0, 0.5):  # 0.5倍的采样率
            _u = 2 * i + 1  # i 是 8*8像素块的横坐标和纵坐标
            for j in np.arange(0.0, 8.0, 0.5):
                _v = 2 * j + 1  # 2*j+1
                index = i * 8 + j
                index1 = int(2 * index - 1)
                for u in range(8):  # u, v is dct系数表
                    for v in range(8):
                        index2 = u * 8 + v
                        t_pi = math.sin(_u * u * math.pi / 16) * (-_u * math.pi / 16) * math.cos(
                            _v * v * math.pi / 16) + math.cos(_u * u * math.pi / 16) * (-_v * math.pi / 16) * math.sin(
                            _v * v * math.pi / 16)
                        t = t_pi * r1 if u == 0 else t_pi * r2  # if u=0, t=t*r1
                        t = t_pi * r1 if v == 0 else t_pi * r2
                        kernel[index1, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, delta_x, delta_y, _table):  # _table是上一个尺度的量化表. delta_x and delta_y
        _kernel = self.kernel * _table  # save from last i_dct layer
        output_x = F.conv2d(input=delta_x, weight=_kernel, stride=1)  # y size:torch.Size([1, 256, 24, 24])
        output_y = F.conv2d(input=delta_y, weight=_kernel, stride=1)
        y = torch.add(output_x, output_y)
        pixel_shuffle = nn.PixelShuffle(2)
        y = pixel_shuffle(y)  # [1,64,48,48]
        return y


class ImplicitTransCompensateForCb(nn.Module):
    def __init__(self, in_channels):
        super(ImplicitTransCompensateForCb, self).__init__()
        self.table = torch.tensor([
            16, 16, 16, 16, 17, 18, 21, 24,
            16, 16, 16, 16, 17, 19, 22, 25,
            16, 16, 17, 18, 20, 22, 25, 29,
            16, 16, 18, 21, 24, 27, 31, 36,
            17, 17, 20, 24, 30, 35, 41, 47,
            18, 19, 22, 27, 35, 44, 54, 65,
            21, 22, 25, 31, 41, 54, 70, 88,
            24, 25, 29, 36, 47, 65, 88, 115]) / 255.0  # .reshape(8, 8)
        self.table = self.table.unsqueeze(-1)  # 设置维数
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)

        temp = torch.empty(256, 1, 1, 1)
        self.factor = nn.Parameter(torch.ones_like(temp))
        self.bias = nn.Parameter(torch.zeros_like(temp))
        self.table = self.table.cuda()

        conv_shape = (256, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')
        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in np.arange(0.0, 8.0, 0.5):  # 0.5倍的采样率
            _u = 2 * i + 1  # i 是 8*8像素块的横坐标和纵坐标
            for j in np.arange(0.0, 8.0, 0.5):
                _v = 2 * j + 1  # 2*j+1
                index = i * 8 + j
                index1 = int(2 * index - 1)
                for u in range(8):  # u, v is dct系数表
                    for v in range(8):
                        index2 = u * 8 + v
                        t_pi = math.sin(_u * u * math.pi / 16) * (-_u * math.pi / 16) * math.cos(
                            _v * v * math.pi / 16) + math.cos(_u * u * math.pi / 16) * (-_v * math.pi / 16) * math.sin(
                            _v * v * math.pi / 16)
                        t = t_pi * r1 if u == 0 else t_pi * r2  # if u=0, t=t*r1
                        t = t_pi * r1 if v == 0 else t_pi * r2
                        kernel[index1, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, x, _table):  # _table是上一个尺度的量化表
        _kernel = self.kernel * _table
        y = F.conv2d(input=x, weight=_kernel, stride=1)  # y size:torch.Size([1, 256, 24, 24])
        pixel_shuffle = nn.PixelShuffle(2)
        y = pixel_shuffle(y)  # [1,64,48,48]
        return y


class ImplicitTransSingleMul(nn.Module):
    def __init__(self, in_channels):
        super(ImplicitTransSingleMul, self).__init__()
        self.table = torch.tensor([
            16, 16, 16, 16, 17, 18, 21, 24,
            16, 16, 16, 16, 17, 19, 22, 25,
            16, 16, 17, 18, 20, 22, 25, 29,
            16, 16, 18, 21, 24, 27, 31, 36,
            17, 17, 20, 24, 30, 35, 41, 47,
            18, 19, 22, 27, 35, 44, 54, 65,
            21, 22, 25, 31, 41, 54, 70, 88,
            24, 25, 29, 36, 47, 65, 88, 115]) / 255.0  # .reshape(8, 8)
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)
        self.table = self.table.unsqueeze(-1)

        self.factor = nn.Parameter(torch.ones_like(self.table))
        self.bias = nn.Parameter(torch.zeros_like(self.table))
        self.table = self.table.cuda()

        conv_shape = (64, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')
        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)

                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, x, weight):
        _table = self.table * self.factor + self.bias
        _kernel = self.kernel * _table
        x = x * weight  # [4, 64, 16, 16])
        y = F.conv2d(input=x, weight=_kernel, stride=1)

        return y


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, use_bias=True, dilation_rate=1):
        super(ConvRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return output


class ConvTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, use_bias=True, dilation_rate=1):
        super(ConvTanh, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.tanh(self.conv(x))
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, use_bias=True, dilation_rate=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)

    def forward(self, x):
        output = self.conv(x)
        return output


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def make_layer_conv(basic_block, num_basic_block):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block)
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=32, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset_mask(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return _ModulatedDeformConv(x, offset, mask, self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups, self.deformable_groups,
                                    self.im2col_step)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):  # 帧间融合
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):  # 5帧的输入
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # aligned change////
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())  # center_frame_idx is 2
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob  # # -1

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class PyramidCell(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(PyramidCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        self.dilation_rate = 0
        # (3, 2, 1, 1, 1, 1)
        self.conv_relu_1 = ConvRelu(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel=3, padding=3,
                                    dilation_rate=dilation_rates[0])
        self.conv_relu_2 = ConvRelu(in_channels=self.in_channels * 2, out_channels=self.out_channels,
                                    kernel=3, padding=2,
                                    dilation_rate=dilation_rates[1])
        self.conv_relu_3 = ConvRelu(in_channels=self.in_channels * 3, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_4 = ConvRelu(in_channels=self.in_channels * 4, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_5 = ConvRelu(in_channels=self.in_channels * 5, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_6 = ConvRelu(in_channels=self.in_channels * 6, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])

    def forward(self, x):
        t = self.conv_relu_1(x)  # 64
        _t = torch.cat([x, t], dim=1)  # 128

        t = self.conv_relu_2(_t)
        _t = torch.cat([_t, t], dim=1)  #

        t = self.conv_relu_3(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_4(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_5(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_6(_t)
        _t = torch.cat([_t, t], dim=1)
        return _t


class DualDomainBlock(nn.Module):  # 0.5 mul sample rate
    def __init__(self, n_channels, n_pyramid_cells, n_pyramid_channels):  # 定义层
        super(DualDomainBlock, self).__init__()
        self.pyramid = PyramidCell(in_channels=n_channels, out_channels=n_pyramid_channels,
                                   dilation_rates=n_pyramid_cells)
        self.conv_1 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_2 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=2, dilation_rate=2)
        self.channel_squeeze = Conv(in_channels=n_channels * 7, out_channels=n_channels,
                                    kernel=1, padding=0)
        self.get_weight_y = GetWeight()
        self.get_weight_c = GetWeight()
        self.implicit_trans_1 = ImplicitTrans(in_channels=n_channels)
        self.implicit_trans_2 = ImplicitTrans(in_channels=n_channels)

        self.pixel_restoration = make_layer(
            ResidualBlockNoBN, 16, num_feat=n_channels
        )  # 此处的resblo去掉
        self.conv_3 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_4 = Conv(in_channels=n_channels * 2, out_channels=n_channels, kernel=3,
                           padding=1)

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):  # x size = [1,64,n,n]
        _t = self.pyramid(x)
        _t = self.channel_squeeze(_t)  # _t torch.Size([1, 64, 24, 24])
        _ty = self.conv_1(_t)
        _tc = self.conv_2(_t)
        _ty = torch.clamp(_ty, -0.5, 0.5)  # 把tensor放在区间内  _ty torch.Size([1, 64, 24, 24])

        ty_weight = self.get_weight_y(_t)
        _ty, _tableY, temp_ty = self.implicit_trans_1(_ty, ty_weight)  # y channel
        tc_weight = self.get_weight_c(_t)
        _tc, _tableC, temp_tc = self.implicit_trans_2(_tc, tc_weight)  # [4,16,16,16]--->[4,256,16,16]  [1,64,48,48]  temp_c is _tc

        _tp = self.pixel_restoration(_t)  # _tp:torch.Size([1, 64, 24, 24])
        _tp = self.conv_3(_tp)  # _tp:torch.Size([1, 64, 24, 24])
        _tp = self.upscale(_tp)

        _td = torch.cat([_ty, _tc], dim=1)  # [4,128,16,16]
        _td = self.conv_4(_td)  # _td:torch.Size([1, 64, 48, 48]) 对用1×1的卷积对y,c 通道进行融合

        y = torch.add(_td, _tp)  # value相加，size不变
        y = y.mul(0.1)

        x = self.upscale(x)
        y = torch.add(x, y)
        return y, _tableY, _tableC, temp_ty, temp_tc


class DualDomainBlockCompensate(nn.Module):
    def __init__(self, n_channels, n_pyramid_cells, n_pyramid_channels):
        super(DualDomainBlockCompensate, self).__init__()
        self.pyramid = PyramidCell(in_channels=n_channels, out_channels=n_pyramid_channels,
                                   dilation_rates=n_pyramid_cells)
        self.conv_1 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_tanh1 = ConvTanh(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_tanh2 = ConvTanh(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_2 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3,
                           padding=2, dilation_rate=2)
        self.channel_squeeze = Conv(in_channels=n_channels * 7, out_channels=n_channels,
                                    kernel=1, padding=0)
        self.get_weight_y = GetWeight()
        self.get_weight_c = GetWeight()
        self.implicit_trans_1 = ImplicitTransCompensate(in_channels=n_channels)
        self.implicit_trans_2 = ImplicitTransCompensateForCb(in_channels=n_channels)

        self.pixel_restoration = make_layer_conv(
            Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1), 1
        )
        self.conv_3 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_4 = Conv(in_channels=n_channels * 2, out_channels=n_channels, kernel=3,
                           padding=1)  # dct改进后in_channel 2--》8

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x, tabley, tablec, temp_ty, temp_tc):  # 利用x估计出delta_x和delta_y [16, 64, 24, 24]
        # x 是上一层IDCT的输出＋步长=2的卷积层， _ty是上一层的输入
        _t = x
        # Y channel compensate
        _ty = temp_ty   # [16, 64, 24, 24] 估计delta_x
        _ty_delta_x = self.conv_tanh1(_ty)
        _ty_delta_y = self.conv_tanh2(_ty)
        # _ty torch.Size([1, 64, 24, 24])
        _ty_delta_x = _ty_delta_x*0.1
        _ty_delta_y = _ty_delta_y*0.1

        # Cb Cr channel
        _tc = temp_tc  # [16, 64, 24, 24]
        # ty_weight = self.get_weight_y(_t)  # 量化表的自适应估计
        _ty = self.implicit_trans_1(_ty_delta_x, _ty_delta_y, tabley)  # compensate
        # tc_weight = self.get_weight_c(_t)
        _tc = self.implicit_trans_2(_tc, tablec)  # [1,64,48,48]

        _tp = self.pixel_restoration(_t)  # [1, 64, 24, 24]
        _tp = self.conv_3(_tp)  # [1, 64, 24, 24]
        _tp = self.upscale(_tp)

        _td = torch.cat([_ty, _tc], dim=1)  # [4,128,16,16]
        _td = self.conv_4(_td)  # [1, 64, 48, 48]

        y = torch.add(_td, _tp)
        y = y.mul(0.1)

        x = self.upscale(x)
        y = torch.add(x, y)
        return y


class DualDomainBlockOrg(nn.Module):
    def __init__(self, n_channels, n_pyramid_cells, n_pyramid_channels):
        super(DualDomainBlockOrg, self).__init__()
        self.pyramid = PyramidCell(in_channels=n_channels, out_channels=n_pyramid_channels,
                                   dilation_rates=n_pyramid_cells)
        self.conv_1 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_2 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3,
                           padding=2, dilation_rate=2)
        self.channel_squeeze = Conv(in_channels=n_channels * 7, out_channels=n_channels,
                                    kernel=1, padding=0)
        self.get_weight_y = GetWeight()
        self.get_weight_c = GetWeight()
        self.implicit_trans_1 = ImplicitTransSingleMul(in_channels=n_channels)
        self.implicit_trans_2 = ImplicitTransSingleMul(in_channels=n_channels)

        self.pixel_restoration = make_layer(
            ResidualBlockNoBN, 16, num_feat=n_channels)  # 此处的resblo去掉
        self.conv_3 = Conv(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_4 = Conv(in_channels=n_channels * 2, out_channels=n_channels, kernel=3,
                           padding=1)

        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        # x size = [1,64,n,n]
        _t = self.pyramid(x)
        _t = self.channel_squeeze(_t)  # _t torch.Size([1, 64, 24, 24])
        _ty = self.conv_1(_t)
        _tc = self.conv_2(_t)
        _ty = torch.clamp(_ty, -0.5, 0.5)  # [4,64,16,16]  _ty torch.Size([1, 64, 24, 24])

        ty_weight = self.get_weight_y(_t)
        _ty = self.implicit_trans_1(_ty, ty_weight)
        tc_weight = self.get_weight_c(_t)
        _tc = self.implicit_trans_2(_tc, tc_weight)  # [4,16,16,16]--->[4,256,16,16]  [1,64,48,48]  换了dataset，导致尺寸变化

        _tp = self.pixel_restoration(_t)  # _tp:torch.Size([1, 64, 24, 24])
        _tp = self.conv_3(_tp)  # _tp:torch.Size([1, 64, 24, 24])

        _td = torch.cat([_ty, _tc], dim=1)  # [4,128,16,16]

        _td = self.conv_4(_td)  # _td:torch.Size([1, 64, 48, 48])

        y = torch.add(_td, _tp)  # value相加，size不变
        y = y.mul(0.1)

        y = torch.add(x, y)
        return y


class VECNN_MF(nn.Module):  # video enhancement CNN network mul frame
    '''
        improve down--->maxpool
    '''
    def __init__(self, n_channels=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=1,
                 with_tsa=True
                 ):
        super(VECNN_MF, self).__init__()
        self.with_tsa = with_tsa
        self.center_frame_idx = center_frame_idx

        # extract features for each frame
        self.conv_first = nn.Conv2d(3, n_channels, 3, 1, 1)

        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=n_channels)
        self.conv_l2_1 = nn.Conv2d(n_channels, n_channels, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(n_channels, n_channels, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1)

        # pcd module
        self.pcd_align = PCDAlignment(
            num_feat=n_channels, deformable_groups=deformable_groups)

        if self.with_tsa:
            self.fusion = TSAFusion(  # 需要改进
                num_feat=n_channels,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * n_channels, n_channels, 1, 1)  # 需要改进

        # reconstruction 重构
        self.reconstruction = make_layer(
            ResidualBlockNoBN, num_reconstruct_block, num_feat=n_channels)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # VE_CNN
        self.n_channels = n_channels
        self.n_pyramids = 1
        self.n_pyramid_cells = [3, 2, 1]
        self.n_pyramid_channels = n_channels

        self.channel_split = nn.Conv2d(in_channels=3, out_channels=n_channels,
                                       kernel_size=5, stride=1, padding=2, bias=False)
        self.downscale_1 = nn.Sequential(  # 下采样的使用2倍采样
            PixelUnshuffle(downscale_factor=2),  # pixel unshuffle的逆操作
            nn.Conv2d(in_channels=n_channels * 2 * 2, out_channels=n_channels,
                      kernel_size=5, stride=1, padding=2, bias=False)
        )
        self.downscale_2 = nn.Sequential(
            PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channels=n_channels * 2 * 2, out_channels=n_channels,
                      kernel_size=5, stride=1, padding=2, bias=False)
        )
        self.upscale_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.upscale_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

        self.conv_relu_X1_1 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)

        # 双域 ×1
        self.dual_domain_blocks_x1 = self.make_layer(
            block=DualDomainBlockOrg,
            num_of_layer=self.n_pyramids)
        self.conv_relu_X1_2 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)

        self.conv_relu_X2_1 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        # ×2块
        self.dual_domain_blocks_x2 = self.make_layer(
            block=DualDomainBlock,
            num_of_layer=self.n_pyramids)
        self.conv_relu_X2_2 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)

        self.conv_relu_X4_1 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        # ×4块
        self.dual_domain_blocks_x4 = self.make_layer(
            block=DualDomainBlock,
            num_of_layer=self.n_pyramids)

        self.conv_relu_X4_2 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)

        self.conv_relu_channel_merge_1 = ConvRelu(in_channels=n_channels * 2, out_channels=n_channels, kernel=3,
                                                  padding=1)
        self.conv_relu_channel_merge_2 = ConvRelu(in_channels=n_channels * 2, out_channels=n_channels, kernel=3,
                                                  padding=1)
        self.conv_relu_output = ConvRelu(in_channels=n_channels, out_channels=3, kernel=5, padding=2)

        self.compensate = DualDomainBlockCompensate(n_channels=64, n_pyramid_cells=[3, 2, 1], n_pyramid_channels=64)
        self.conv_compensate_X4 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1)
        self.conv_compensate_X2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(n_channels=self.n_channels, n_pyramid_cells=self.n_pyramid_cells,
                                n_pyramid_channels=self.n_pyramid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, (
            'The height and width must be multiple of 4.')

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)

        # TSA fusion
        t_x1 = self.fusion(aligned_feat)  # [4,64,96,96]

        maxpooldownscale = maxpooldown()
        t_x2 = maxpooldownscale(t_x1)  # [4,64,48,48]
        t_x4 = maxpooldownscale(t_x2)  # [4,64,24,24]

        t_x4 = self.conv_relu_X4_1(t_x4)  # [4,64,24,24]
        t_x4, t_x4_table_y, t_x4_table_c, temp_ty, temp_tc = self.dual_domain_blocks_x4(t_x4)  # [4,64,48,48]
        t_x4 = self.conv_relu_X4_2(t_x4)  # [4,64,48,48]

        t_x2_temp = torch.cat((t_x2, t_x4), 1)  # [1, 64, 48, 48]
        t_x2_temp = self.conv_relu_channel_merge_1(t_x2_temp)  # [1, 64, 48 ,48]
        # compensate
        t_x2_temp = self.conv_compensate_X4(t_x2_temp)
        t_x2_comp = self.compensate(t_x2_temp, t_x4_table_y, t_x4_table_c, temp_ty, temp_tc)
        t_x2 = torch.cat((t_x2, t_x2_comp), 1)
        t_x2 = self.conv_relu_channel_merge_1(t_x2)  # [1, 64, 48, 48]

        t_x2 = self.conv_relu_X2_1(t_x2)  # [1, 64, 48, 48]
        t_x2, t_x2_table_y, t_x2_table_c, temp_t2y, temp_t2c = self.dual_domain_blocks_x2(t_x2)  # [1, 64, 96, 96]
        t_x2 = self.conv_relu_X2_2(t_x2)  # [1,64,96,96]

        t_x1_temp = torch.cat((t_x1, t_x2), 1)  # [1,128,96,96]
        t_x1_temp = self.conv_relu_channel_merge_2(t_x1_temp)  # [1,64,96,96]
        # compensate
        t_x1_temp = self.conv_compensate_X2(t_x1_temp)
        t_x4_comp = self.compensate(t_x1_temp, t_x2_table_y, t_x2_table_c, temp_t2y, temp_t2c)  # [1,64,48,48]
        t_x1 = torch.cat((t_x1, t_x4_comp), 1)  # [1,128,96,96]
        t_x1 = self.conv_relu_channel_merge_2(t_x1)  # [1,128,96,96]

        t_x1 = self.conv_relu_X1_1(t_x1)  # [1,64,96,96]
        t_x1 = self.dual_domain_blocks_x1(t_x1)  # [1,64,96,96]
        t_x1 = self.conv_relu_X1_2(t_x1)  # [1,64,96,96]

        # FcaNet 可有可无
        # attention_block = MultiSpectralAttentionLayer(64, 16, 16)
        # output = attention_block(t_x1)
        # y = self.conv_relu_output(output)  # [4,3,96,96]

        return y  # pth_size < 60MB