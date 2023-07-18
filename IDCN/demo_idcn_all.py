from __future__ import print_function
import argparse
from os import listdir
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from dataset import *
from skimage.metrics import structural_similarity as ssim
from utils_idcn import *

# Training settings
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--quality', default=10, type=int, help='the qf of jpeg')
parser.add_argument('--input_LR_path', type=str, default='datasets_test/hevc_sequence_qp_27/', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='datasets_test/hevc_sequence_gt_frames/gt/', help='input path to use')
parser.add_argument('--model', type=str, default='/home/datasets/zhr/IDCN_Pytorch/checkpoints/qp37/current_model.pth', help='model file to use')
parser.add_argument('--output_path', default='result/IDCN/hevc_sequence_qp_27/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--calc_on_y', default=False, action='store_true', help='calc on y channel')
opt = parser.parse_args()


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, input_tensor)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def padding8(img):
    h, w = img.shape[0:2]
    pad_h = 8 - h % 8 if h % 8 != 0 else 0
    pad_w = 8 - w % 8 if w % 8 != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'edge')
    return img


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


loader = transforms.Compose([
    transforms.ToTensor()])

path = opt.input_LR_path
path_HR = opt.input_HR_path

# image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
# print(image_nums, 'test images')

sigma = get_sigma_c1(opt.quality)

input_root_path = '../datasets_test/hevc_sequence_qp_27/'
output_root_path = 'results/IDCN/hevc_sequence_qp_27/'
gt_root_path = '../datasets_test/hevc_sequence_gt_frames/'
folder_list = sorted(listdir(input_root_path))

for sequence in folder_list:  # A B C D E
    sub_folder_list = sorted(listdir('{}/{}'.format(input_root_path, sequence)))
    for video in sub_folder_list:  # PeopleOnStreet_2560x1600_30_crop Traffic_2560x1600_30_crop
        input_path = '{}{}/{}/'.format(input_root_path, sequence, video)
        output_path = '{}{}/{}/'.format(output_root_path, sequence, video)
        gt_path = '{}{}/{}/'.format(gt_root_path, sequence, video)
        img_list = []
        for frame in listdir(input_path):
            img_list.append(listdir(input_path))
        print(input_path, ' ', len(img_list), 'test images')
        input_psnr_avg = 0
        output_psnr_avg = 0
        input_ssim_avg = 0
        output_ssim_avg = 0
        index = 0
        for i in listdir(input_path):
            if is_image_file(i):
                with torch.no_grad():
                    index = index + 1

                    input_img = cv.imread('{}{}.png'.format(input_path, "%03d" % index))

                    input_tensor = transforms.ToTensor()(input_img)
                    input_tensor = torch.unsqueeze(input_tensor, dim=0).float()

                    input_with_label = np.concatenate([input_img,
                                                       sigma[0:input_img.shape[0], 0:input_img.shape[1], :]], axis=-1) #4 dimes
                    input = NumpyToTensor()(input_with_label)

                    input = torch.unsqueeze(input, dim=0).float()

                    model = torch.load(opt.model, map_location='cuda:0')
                    model.eval()
                    if opt.cuda:
                        model = model.cuda()
                        input = input.cuda()

                    out = model(input)
                    out = out.cpu()

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    if not os.path.exists('{}output/'.format(output_path)):
                        os.makedirs('{}output/'.format(output_path))
                    if not os.path.exists('{}gt/'.format(output_path)):
                        os.makedirs('{}gt/'.format(output_path))
                    if not os.path.exists('{}input/'.format(output_path)):
                        os.makedirs('{}input/'.format(output_path))

                    save_image_tensor(out, '{}output/{}.png'.format(output_path, "%03d" % index))

                    img_original = cv.imread('{}{}.png'.format(gt_path, "%03d" % index))
                    cv.imwrite('{}gt/{}.png'.format(output_path, "%03d" % index), img_original)

                    input_center_img = cv.imread('{}{}.png'.format(input_path, "%03d" % index))
                    cv.imwrite('{}input/{}.png'.format(output_path, "%03d" % index), input_center_img)

                    out = out.squeeze(0)
                    out_img = out.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(
                        torch.uint8).numpy()

                    if opt.calc_on_y is True:
                        input_center_img = cv.cvtColor(input_center_img, cv.COLOR_BGR2YUV)
                        input_center_img = input_center_img[:, :, 0]
                        img_original = cv.cvtColor(img_original, cv.COLOR_BGR2YUV)
                        img_original = img_original[:, :, 0]
                        out_img = cv.cvtColor(out_img, cv.COLOR_BGR2YUV)
                        out_img = out_img[:, :, 0]

                    input_psnr = calc_psnr(transforms.ToTensor()(input_center_img),
                                           transforms.ToTensor()(img_original))
                    output_psnr = calc_psnr(transforms.ToTensor()(out_img),
                                            transforms.ToTensor()(img_original))

                    input_ssim = ssim(input_center_img,
                                      img_original, multichannel=True)
                    output_ssim = ssim(out_img,
                                       img_original, multichannel=True)

                    print('input_psnr', input_psnr)
                    print('output_psnr', output_psnr)
                    print('input_ssim', input_ssim)
                    print('output_ssim', output_ssim)

                    input_psnr_avg += input_psnr
                    output_psnr_avg += output_psnr
                    input_ssim_avg += input_ssim
                    output_ssim_avg += output_ssim
        input_psnr_avg = input_psnr_avg / index
        output_psnr_avg = output_psnr_avg / index
        psnr_increase = output_psnr_avg - input_psnr_avg
        input_ssim_avg = input_ssim_avg / index
        output_ssim_avg = output_ssim_avg / index
        ssim_increase = output_ssim_avg - input_ssim_avg
        # print('input_psnr_avg:', input_psnr_avg)
        # print('output_psnr_avg:', output_psnr_avg)
        print('psnr_increase:', psnr_increase)
        # print('input_ssim_avg:', "%.4f" % input_ssim_avg)
        # print('output_ssim_avg:', "%.4f" % output_ssim_avg)
        print('ssim_increase:', "%.4f" % ssim_increase)
        with open("results/IDCN/result.txt", "a") as f:
            f.write(input_path)
            f.write('\npsnr_increase:{}'.format(psnr_increase))
            f.write('\nssim_increase:{}\n'.format(ssim_increase))
