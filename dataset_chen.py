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
        num, r, c, ch = lr.shape

        if hflip:
            hr = hr[:, ::-1, :]
            for idx in range(num):
                lr[idx, :, :, :] = lr[idx, :, ::-1, :]
        if vflip:
            hr = hr[::-1, :, :]
            for idx in range(num):
                lr[idx, :, :, :] = lr[idx, ::-1, :, :]
        if rot90:
            hr = hr.transpose(1, 0, 2)
            lr = lr.transpose(0, 2, 1, 3)

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

        h, w = lr.shape[1: 3]
        # print(h, w)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_lr = lr[:, top:top + new_h, left: left + new_w, :]
        new_hr = hr[top * self.scale:top * self.scale + new_h * self.scale,
                 left * self.scale: left * self.scale + new_w * self.scale, :]

        return {'lr': new_lr, "hr": new_hr, "im_name": name}


class ToTensor(object):
    def __call__(self, sample):
        lr, hr, name = sample['lr'] / 255.0, sample['hr'] / 255.0, sample['im_name']
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1), 'im_name': name}


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
        self.dir_LR = '{}low_gt/'.format(dir)  # low_gt
        self.lis = sorted(os.listdir(self.dir_HR))
        self.crop_size = 64
        self.scale = 1  # 4--》1
        self.transform = transforms.Compose([RandomCrop(self.crop_size, self.scale),
                                             DataAug(),
                                             ToTensor()])

    def __len__(self):
        return len(self.lis)

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

        frames_lr = np.zeros((5, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - 2, center_index + 3):  # only use 5 frames
            i = j - center_index + 2
            frames_lr_name = '{}{}/{}'.format(self.dir_LR, folder_name, img_list[j])
            img = cv.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr, 'im_name': img_list[center_index]}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']


class ValidateDataset(data.Dataset):
    def __init__(self, dir):
        self.dir_HR = '{}sequence_gt/'.format(dir)
        self.dir_LR = '{}sequence_cbr_high/'.format(dir)
        self.lis = sorted(os.listdir('{}/'.format(self.dir_HR)))

        self.scale = 1
        self.transform = transforms.Compose([ToTensor()])

    def __len__(self):
        return len(self.lis)

    def __getitem__(self, idx):
        folder_list = sorted(listdir(self.dir_HR))
        folder_index = random.randint(0, (len(folder_list) - 1))
        folder_name = folder_list[folder_index]
        folder_path = '{}{}/'.format(self.dir_HR, folder_name)  # 随机的文件夹

        img_list = sorted(listdir(folder_path))  # 读取文件夹里的图片地址

        # get frame size
        image_example_name = '{}{}'.format(folder_path, img_list[0])  # 输出是文件夹中图片的地址

        image_example = cv.imread(image_example_name)
        h, w, ch = image_example.shape

        # if 3 <= idx <= (len(img_list) - 3):
        #     center_index = idx
        # else:
        #     center_index = 0
        center_index = 10
        
        frames_hr_name = '{}{}'.format(folder_path, img_list[center_index])
        frames_hr = cv.imread(frames_hr_name)

        frames_lr = np.zeros((5, int(h / self.scale), int(w / self.scale), ch))
        for j in range(center_index - 2, center_index + 3):  # only use 5 frames
            i = j - center_index + 2
            frames_lr_name = '{}/{}/{}'.format(self.dir_LR, folder_name, img_list[j])
            img = cv.imread(frames_lr_name)
            frames_lr[i, :, :, :] = img  # t h w c

        sample = {'lr': frames_lr, 'hr': frames_hr, 'im_name': img_list[center_index]}
        sample = self.transform(sample)

        return sample['lr'], sample['hr']
        
        
class ValidateDataset_zhao(data.Dataset):
    def __init__(self, dir):
        self.dir_HR = '{}sequence_gt/'.format(dir)
        self.dir_LR = '{}sequence_cbr_high/'.format(dir)
        self.folder_list = sorted(listdir(self.dir_HR))  # list 是gt_frame下的文件夹
        self.img_list = []
        for _f in self.folder_list:
            target_folder_path = '{}{}/'.format(self.dir_HR, _f)  # datasets/validate/gt_frame/200 || 100||
            self.img_list.append(sorted(listdir(target_folder_path)))  # datasets/validate/gt_frame/190-----200 十个路径

    def __len__(self):
        return 180

    def __getitem__(self, idx):
        folder_index = idx // 20
        
        img_index = idx
        if idx <= 3:
            img_index = 2
        if idx >= (len(self.img_list) - 3):  # len(img_list) = 10
            img_index = len(self.img_list) - 3
        # floder_index 是 idx//20, img_index 是 计算而来
        target_img_name = '{}{}/{:0>3d}.png'.format(self.dir_HR, self.folder_list[folder_index],
                                                    (img_index % 20) + 1)
        

        target_img = cv.imread(target_img_name)
        h, w, ch = target_img.shape

        input_img = np.zeros((5, h, w, ch))
        input_img_name_1 = '{}{}/{:0>3d}.png'.format(self.dir_LR, self.folder_list[folder_index], img_index - 2 + 1)
        input_img[0, :, :, :] = cv.imread(input_img_name_1)
        input_img_name_2 = '{}{}/{:0>3d}.png'.format(self.dir_LR, self.folder_list[folder_index], img_index - 1 + 1)
        input_img[1, :, :, :] = cv.imread(input_img_name_2)
        input_img_name_3 = '{}{}/{:0>3d}.png'.format(self.dir_LR, self.folder_list[folder_index], img_index + 1)
        input_img[2, :, :, :] = cv.imread(input_img_name_3)
        input_img_name_4 = '{}{}/{:0>3d}.png'.format(self.dir_LR, self.folder_list[folder_index], img_index + 1 + 1)
        input_img[3, :, :, :] = cv.imread(input_img_name_4)
        input_img_name_5 = '{}{}/{:0>3d}.png'.format(self.dir_LR, self.folder_list[folder_index], img_index + 2 + 1)
        input_img[4, :, :, :] = cv.imread(input_img_name_5)

        target_tensor = NumpyToTensor()(target_img.copy())  # [3, 128, 128]
        input_tensor = NumpyToTensor(multi_frame=True)(input_img.copy())  # [5, 3, 128, 128]
        return input_tensor, target_tensor
