import os
from os import listdir
import random
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2 as cv

current_folder_index = 0
current_img_index = 2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def get_training_set():
    root_path = 'datasets/train/'
    return TrainDataset(dir=root_path)


def get_validate_set():
    root_path = 'datasets/validate/'
    return ValidateDataset(dir=root_path)


class DataAug(object):
    def __call__(self, sample):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        lr, hr, name = sample['lr'], sample['hr'], sample['im_name']
        r, c, ch = lr.shape

        if hflip:
            hr = hr[:, ::-1, :]
            lr[:, :, :] = lr[:, ::-1, :]
        if vflip:
            hr = hr[::-1, :, :]
            lr[:, :, :] = lr[::-1, :, :]
        if rot90:
            hr = hr.transpose(1, 0, 2)
            lr = lr.transpose(1, 0, 2)

        return {'lr': lr, 'hr': hr, 'im_name': name}


class RandomCrop(object):
    def __init__(self, output_size, scale):
        self.scale = scale
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        lr, hr, name = sample['lr'], sample['hr'], sample['im_name']

        h = lr.shape[0]
        w = lr.shape[1]
        new_h, new_w = self.output_size  # 64, 64

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_lr = lr[top:top + new_h, left: left + new_w, :]
        new_hr = hr[top:top + new_h, left: left + new_w, :]

        return {'lr': new_lr, "hr": new_hr, "im_name": name}


class ToTensor(object):
    def __call__(self, sample):
        lr, hr, name = sample['lr'] / 255.0, sample['hr'] / 255.0, sample['im_name']
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(2, 0, 1), 'hr': hr.permute(2, 0, 1), 'im_name': name}


class NumpyToTensor(object):
    def __init__(self, multi_frame=False):
        self.multi_frame = multi_frame

    def __call__(self, numpy_input):
        numpy_input = numpy_input / 255.0
        numpy_input = torch.from_numpy(numpy_input).float()
        if self.multi_frame is True:
            return numpy_input.permute(0, 3, 1, 2)
        else:
            return numpy_input.permute(2, 0, 1)


class TrainDataset(data.Dataset):
    def __init__(self, dir):
        self.dir_HR = '{}gt/'.format(dir)  # gt
        self.dir_LR = '{}ntire_cbr_600/'.format(dir)  # low_gt
        self.lis = sorted(os.listdir(self.dir_HR))
        self.crop_size = 64
        self.scale = 1  # 4--》1
        self.transform = transforms.Compose([RandomCrop(self.crop_size, self.scale),
                                             DataAug(),
                                             ToTensor()])

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        folder_list = sorted(listdir(self.dir_HR))
        folder_index = random.randint(0, (len(folder_list) - 1))
        folder_name = folder_list[folder_index]
        folder_path = '{}{}/'.format(self.dir_HR, folder_name)

        img_list = sorted(listdir(folder_path))
        # get frame size
        image_example_name = '{}{}'.format(folder_path, img_list[0])
        image_example = cv.imread(image_example_name)
        h, w, ch = image_example.shape  # gt的shape

        center_index = random.randint(2, len(img_list) - 3)

        frames_hr_name = '{}{}'.format(folder_path, img_list[center_index])
        frames_hr = cv.imread(frames_hr_name)

        frames_lr = np.zeros((int(h / self.scale), int(w / self.scale), ch))
        # for j in range(center_index - 2, center_index + 3):  # only use 5 frames
        #     i = j - center_index + 2
        #     frames_lr_name = '{}{}/{}'.format(self.dir_LR, folder_name, img_list[j])
        #     img = cv.imread(frames_lr_name)
        #     frames_lr[i, :, :, :] = img  # t h w c
        frames_lr_name = '{}{}/{}'.format(self.dir_LR, folder_name, img_list[center_index])
        img = cv.imread(frames_lr_name)
        frames_lr[:, :, :] = img

        sample = {'lr': frames_lr, 'hr': frames_hr, 'im_name': img_list[center_index]}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']


class ValidateDataset(data.Dataset):  # validate需要修改,验证集不用随机数
    ''' folder_index 6
    target_img_name datasets / validate / sequence_gt / Keiba_416x240_30 / 007.png
    folder_index 1
    target_img_name datasets / validate / sequence_gt / BasketballDrill_832x480_50 / 007.png
    folder_index 0
    target_img_name datasets / validate / sequence_gt / BQSquare_416x240_60 / 003.png
    folder_index 4
    target_img_name datasets / validate / sequence_gt / FourPeople_1280x720_60 / 007.png
    '''

    def __init__(self, dir):
        self.target_path = '{}ntire_gt/'.format(dir)
        self.input_path = '{}ntire_cbr_600/'.format(dir)
        self.folder_list = sorted(listdir(self.target_path))
        self.img_list = []
        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.target_path, _f)
            self.img_list.append(sorted(listdir(target_folder_path)))

    def __len__(self):
        return 100  # 表示val次数，==20

    def __getitem__(self, idx):
        folder_index = idx // 20
        img_index = idx
        if idx <= 3:
            img_index = 2
        if idx >= (len(self.img_list) - 3):
            img_index = len(self.img_list) - 3
        target_img_name = '{}{}/{:0>3d}.png'.format(self.target_path, self.folder_list[folder_index],
                                                    (img_index % 20) + 1)  # 一张图片

        target_img = cv.imread(target_img_name)
        h, w, ch = target_img.shape

        input_img = np.zeros((h, w, ch))  # 1 picture validate
        input_img_name_1 = '{}{}/{:0>3d}.png'.format(self.input_path, self.folder_list[folder_index],
                                                     (img_index % 20) + 1)
        input_img[:, :, :] = cv.imread(input_img_name_1)

        target_tensor = NumpyToTensor()(target_img.copy())  # [3, 128, 128] gt copy
        input_tensor = NumpyToTensor(multi_frame=False)(input_img.copy())  # [3, 128, 128]
        return input_tensor, target_tensor

