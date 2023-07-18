import argparse
import os
import sys
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import utils as vutils
from utils_idcn import *

from dataset_idcn import *
from rootmodel.IDCN import *


parser = argparse.ArgumentParser(description="PyTorch VECNN")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--dataset_target", default='datasets/train/gt/', type=str, help="dataset  path")
parser.add_argument("--dataset_input", default='datasets/train/ntire_cbr_600/', type=str,
                    help="dataset path")
parser.add_argument("--dataset_valid_gt", default='datasets/validate/ntire_gt/', type=str,
                    help="dataset path")
parser.add_argument("--dataset_valid_input", default='datasets/validate/ntire_cbr_600/', type=str,
                    help="dataset path")
parser.add_argument("--checkpoints_path", default='checkpoints/IDCN/', type=str, help="checkpoints path")
parser.add_argument("--resume", default='checkpoints/IDCN/best_model.pth', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=50,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=200")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
parser.add_argument("--device", default='0', type=str, help="which gpu to use")
parser.add_argument("--visualization", default='wandbd', type=str, help="none or wandb or visdom")
opt = parser.parse_args()

min_avr_loss = 99999999
epoch_avr_loss = 0
n_iter = 0
psnr_avr = 0
last_psnr_avr_1 = 0
last_psnr_avr_2 = 0
last_psnr_avr_3 = 0
last_psnr_avr_4 = 0
last_psnr_avr_5 = 0
psnr_max = 0
lr = opt.lr
adjust_lr_flag = False
stop_flag = False
save_flag = True
opt.resume = False


def main():
    global opt, model
    global stop_flag
    global optimizer

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device

    if opt.visualization == 'wandb':
        wandb.init(project="CBREN")

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = get_training_set()
    # train_set = TrainDatasetMultiFrame(target_path=opt.dataset_target, input_path=opt.dataset_input)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=opt.threads)

    valid_set = get_validate_set()
    # valid_set = ValidateDataset(target_path=opt.dataset_valid_gt, input_path=opt.dataset_valid_input)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=1,
                                   shuffle=False, num_workers=opt.threads)

    print("===> Building model")
    pyramid_cells = (3, 2, 1, 1, 1, 1)
    qy = get_table(luminance_quant_table, opt.qf)
    qc = get_table(chrominance_quant_table, opt.qf)
    model = IDCN(n_channels=64, n_pyramids=8,
                 n_pyramid_cells=pyramid_cells, n_pyramid_channels=64, qy=qy, qc=qc)

    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        # model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()


    if opt.visualization == 'wandb':
        wandb.watch(model)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint.state_dict(), strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if stop_flag is True:
            print('finish, training terminate!!!')
            break
        train(training_data_loader, optimizer, model, criterion, epoch)
        valid(valid_data_loader, model, epoch)
        save_checkpoint(model, epoch, optimizer)


def adjust_learning_rate(epoch):
    global lr
    global adjust_lr_flag
    global stop_flag
    # if adjust_lr_flag is True:
    #     lr = lr * 0.5
    #     if lr < 1e-6:
    #         stop_flag = True
    #     print('-------------adjust lr to [{:.7}]-------------'.format(lr))
    #     adjust_lr_flag = False
    lr = opt.lr * (0.1 ** (epoch // opt.step))

    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter
    global psnr_avr

    avr_loss = 0

    lr = adjust_learning_rate()

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target = batch[0], batch[1]  # b c h w

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        out = model(input)

        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        avr_loss += loss.item()
        sys.stdout.write("===> Epoch[{}]({}/{}): Loss: {:.6f}\r".
                         format(epoch, iteration, len(training_data_loader), loss.item()))

    avr_loss = avr_loss / len(training_data_loader)

    if opt.visualization == 'wandb':
        wandb.log({'Loss': avr_loss, 'Valid PSNR': psnr_avr, 'Learning Rate': lr})

    epoch_avr_loss = avr_loss
    print('\nepoch_avr_loss[{:.6f}]'.format(epoch_avr_loss))


def save_checkpoint(model, epoch, optimizer):
    global min_avr_loss
    global save_flag

    model_folder = opt.checkpoints_path
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if (epoch % 5) == 0:
        torch.save(model, model_folder + "model_epoch_{}.pth".format(epoch))
        print("Checkpoint saved to {}".format(model_folder))
    if save_flag is True:
        torch.save(model, model_folder + "best_model.pth")
        save_flag = False
    torch.save(model, model_folder + "current_model.pth")


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def valid(valid_data_loader, model, epoch):
    global adjust_lr_flag
    global psnr_avr
    global last_psnr_avr_1
    global last_psnr_avr_2
    global last_psnr_avr_3
    global last_psnr_avr_4
    global last_psnr_avr_5
    global psnr_max
    global save_flag
    last_psnr_avr_5 = last_psnr_avr_4
    last_psnr_avr_4 = last_psnr_avr_3
    last_psnr_avr_3 = last_psnr_avr_2
    last_psnr_avr_2 = last_psnr_avr_1
    last_psnr_avr_1 = psnr_avr
    if (epoch % 1) == 0:
        psnr_sum = 0
        sys.stdout.write('valid processing\r')
        with torch.no_grad():
            for iteration, batch in enumerate(valid_data_loader, 1):
                input, target = batch[0], batch[1]
                model.eval()
                if opt.cuda:
                    model = model.cuda()
                    input = input.cuda()

                output = model(input)
                output = output.cpu()

                psnr_sum += calc_psnr(target, output).item()

        psnr_avr = psnr_sum / (len(listdir(opt.dataset_valid_gt)) * 20)
        if psnr_max < psnr_avr:
            psnr_max = psnr_avr
            save_flag = True
            print('||||||||||||||||||||||best psnr is[{:.6f}]||||||||||||||||||||||'.format(psnr_max))
        if last_psnr_avr_1 < last_psnr_avr_5 and last_psnr_avr_2 < last_psnr_avr_5 and last_psnr_avr_3 < last_psnr_avr_5 \
                and last_psnr_avr_4 < last_psnr_avr_5 and psnr_avr < last_psnr_avr_5 and epoch > 1:
            adjust_lr_flag = True
            last_psnr_avr_1 = 0
            last_psnr_avr_2 = 0
            last_psnr_avr_3 = 0
            last_psnr_avr_4 = 0
            last_psnr_avr_5 = 0
        print('psnr_valid:[{:.6f}],last_psnr_avr_1:[{:.6f}],last_psnr_avr_2:[{:.6f}],last_psnr_avr_3:[{:.6f}],'
              'last_psnr_avr_4:[{:.6f}],last_psnr_avr_5:[{:.6f}]'
              .format(psnr_avr, last_psnr_avr_1, last_psnr_avr_2,
                      last_psnr_avr_3, last_psnr_avr_4, last_psnr_avr_5))


if __name__ == "__main__":
    main()
