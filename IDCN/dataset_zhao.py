import os
import random
from os import listdir
import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from random import randint
from utils import get_sigma_c1


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class DataExpasion(object):
    def __call__(self, img_input):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip:
            img_input = img_input[:, ::-1, :]
        if vflip:
            img_input = img_input[::-1, :, :]
        if rot90:
            img_input = img_input.transpose(1, 0, 2)
        return img_input


class TwinDataExpasion(object):
    def __call__(self, img_input):
        img_input_1 = img_input[0]
        img_input_2 = img_input[1]
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        if hflip:
            img_input_1 = img_input_1[:, ::-1, :]
            img_input_2 = img_input_2[:, ::-1, :]
        if vflip:
            img_input_1 = img_input_1[::-1, :, :]
            img_input_2 = img_input_2[::-1, :, :]
        if rot90:
            img_input_1 = img_input_1.transpose(1, 0, 2)
            img_input_2 = img_input_2.transpose(1, 0, 2)
        return [img_input_1, img_input_2]


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img_input):
        h, w = img_input.shape[0], img_input.shape[1]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_input = img_input[top:top + new_h, left: left + new_w, :]
        return new_input


class TwinRandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img_input):
        img_input_1 = img_input[0]
        img_input_2 = img_input[1]
        h = img_input_1.shape[0]
        w = img_input_1.shape[1]
        new_h = self.output_size[0]
        new_w = self.output_size[1]

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_input_1 = img_input_1[top:top + new_h, left: left + new_w, :]
        new_input_2 = img_input_2[top:top + new_h, left: left + new_w, :]
        return [new_input_1, new_input_2]


class NumpyToTensor(object):
    def __call__(self, numpy_input):
        numpy_input = numpy_input / 255.0
        numpy_input = torch.from_numpy(numpy_input).float()
        return numpy_input.permute(2, 0, 1)


class TrainDataset(data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data_path = 'datasets/train/'
        self.lis = sorted(os.listdir('{}gt/'.format(self.data_path)))
        self.crop_size = 128
        self.quality = 10
        # self.quality = random.randint(1, 60)
        # self.transform = transforms.Compose([
        #     RandomCrop(self.crop_size),
        #     DataExpasion(),
        # ])
        self.twin_transform = transforms.Compose([
            TwinRandomCrop(self.crop_size),
            TwinDataExpasion(),
        ])
        self.to_tensor = NumpyToTensor
        self.sigma = get_sigma_c1(self.quality)

    def __getitem__(self, index):
        target_path = '{}gt/{}'.format(self.data_path, self.lis[index])
        input_path = '{}qf_{}/{}'.format(self.data_path, self.quality, self.lis[index])
        target_img = cv.imread(target_path)
        input_img = cv.imread(input_path)
        sample = [target_img, input_img]
        sample = self.twin_transform(sample)
        target_img = sample[0]
        input_img = sample[1]
        input_with_label = np.concatenate([input_img,
                                           self.sigma[0:input_img.shape[0], 0:input_img.shape[1], :]], axis=-1)

        target_img = transforms.ToTensor()(target_img.copy())
        input_with_label = self.to_tensor()(input_with_label.copy())
        return input_with_label, target_img

    def __len__(self):
        return 60000


class DatasetFromFolder(data.Dataset):
    def __init__(self, target_path, input_path):
        self.target_path = target_path
        self.input_path = input_path
        self.folder_list = sorted(listdir(self.target_path))
        self.img_list = []
        self.quality = 10
        self.to_tensor = NumpyToTensor
        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.target_path, _f)
            self.img_list.append(sorted(listdir(target_folder_path)))

        self.crop_size = 96
        # self.crop_size = 128
        self.twin_transform = transforms.Compose([
            TwinRandomCrop(self.crop_size),
            TwinDataExpasion()
        ])
        self.sigma = get_sigma_c1(self.quality)

    def __len__(self):
        # return len(self.folder_list)
        # return 1000
        return 60000

    def __getitem__(self, idx):
        folder_index = randint(0, (len(self.folder_list) - 1))
        img_index = randint(2, len(self.img_list[folder_index]) - 3)
        target_img_name = '{}{}/{}.png'.format(self.target_path, self.folder_list[folder_index],
                                               img_index)

        target_img = cv.imread(target_img_name)

        input_img_name = '{}{}/{}.png'.format(self.input_path, self.folder_list[folder_index], img_index)
        input_img = cv.imread(input_img_name)
        sample = [target_img, input_img]
        sample = self.twin_transform(sample)
        target_img = sample[0]
        input_img = sample[1]
        input_img = np.concatenate([input_img,
                                    self.sigma[0:input_img.shape[0], 0:input_img.shape[1], :]], axis=-1)

        target_tensor = transforms.ToTensor()(target_img.copy())  # [3, 128, 128]
        input_tensor = self.to_tensor()(input_img.copy())  # [5, 3, 128, 128]
        return input_tensor, target_tensor
