import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import VECNN_MF
from dataset import *
import visdom
import wandb
import numpy as np
import os

import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

torch.cuda.set_device(0)  # use the chosen gpu
# os.environ['CUDA_VISIBLE_DEVICES']='2'


none_or_wandb_or_visdom = 2  # 0 is none, 1 is wandb, 2 is visdom

if none_or_wandb_or_visdom == 1:
    wandb.init(project="CBREN")
elif none_or_wandb_or_visdom == 2:
    vis = visdom.Visdom(env='CBREN_2')


# setting eval_mini
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / (torch.mean((img1 - img2) ** 2)))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Training settings
parser = argparse.ArgumentParser(description="PyTorch EDVR")
parser.add_argument("--dataset", default='datasets/', type=str, help="dataset path")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=5000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-04, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--test_path", default='dataset/test/', type=str)
parser.add_argument("--validate_path", default='datasets/validate/', type=str)
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")

opt = parser.parse_args()
min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0
max_indicator = 0
fail_count = 0

# wandb setting
wandb.init(
    project="CBREN", name="original_GPU_10",
    config={
        "lr_rate": 1e-4,
        "epoch": 10000,
        "batch_size": 8
    }
)


class CharbonnierLoss(torch.nn.Module):

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # random number
    opt.seed = random.randint(1, 10000)
    # print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = get_training_set()
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True,
                                      num_workers=opt.threads, drop_last=True)

    print("===> Building model")
    model = VECNN_MF()
    print("n_pyramids: " + str(model.n_pyramids) + " " + "n_pyramid_cells: " + str(model.n_pyramid_cells))

    criterion = CharbonnierLoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # if none_or_wandb_or_visdom == 1:
    wandb.watch(model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 0
            model.load_state_dict(checkpoint.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    print("===> Training")
    psnr = 0
    lr = opt.lr
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if epoch == 0:
            new_lr = opt.lr
        else:
            new_lr = lr
        lr = adjust_learning_rate(epoch - 1, psnr, new_lr)
        train(training_data_loader, optimizer, model, criterion, epoch, lr)
        psnr = validate(model, optimizer, epoch)
        save_checkpoint(model, epoch)


def adjust_learning_rate(epoch, psnr, lr):
    """Sets the learning rate to the initial LR decayed by 10"""
    # lr = opt.lr   学习率随着epoch下降 , 初始值是e-4 , 10*(-4)
    # lr = opt.lr * (0.5 ** (epoch // opt.step))
    global max_indicator
    global fail_count

    _loss = psnr
    # update learning rate
    if epoch % 50 == 0 and epoch > 1400:
        if _loss > max_indicator:
            if _loss - max_indicator < 1e-3:
                fail_count += 1
            else:
                fail_count = 0
            max_indicator = _loss
        else:
            fail_count += 1
        # when fail count > 4
        if fail_count >= 4:
            if lr > 2e-6:
                fail_count = 0
                lr = lr / 5
                lr = max(1e-6, lr)

    print('lr:{}  iter:{}'.format(lr, n_iter))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, lr):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter

    avr_loss = 0

    # lr = adjust_learning_rate(optimizer, epoch - 1)  # 如何连续五个epoch指标不变，lr降到一半

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target = batch[0], batch[1]  # input: b, t, c, h, w target: t, c, h, w

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avr_loss += loss.item()

        # if iteration % 100 == 0:
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                            loss.item()))
    avr_loss = avr_loss / len(training_data_loader)
    wandb.log({'epoch': epoch, 'avg_loss': avr_loss})

    epoch_avr_loss = avr_loss
    if epoch_avr_loss < min_avr_loss:
        min_avr_loss = epoch_avr_loss
        print('|||||||||||||||||||||min_epoch_loss is {:.10f}|||||||||||||||||||||'.format(min_avr_loss))
        save_flag = True
    else:
        save_flag = False
        print('epoch_avr_loss is {:.10f}'.format(epoch_avr_loss))


def save_checkpoint(model, epoch):
    global min_avr_loss
    global save_flag

    model_folder = "checkpoints/original/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    # state = {"epoch": epoch, "model": model}
    # 每50个epoch保存模型
    if (epoch % 50) == 0:
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    if save_flag is True:
        torch.save(model, '{}epoch_{}_min_batch_loss_{}.pth'.format(model_folder, epoch, min_avr_loss))
        print('min_loss model saved')


def test(present_model, optimizer, epoch_num):
    print("----Start Test----")
    psnr = AverageMeter()
    test_set = get_test_set()
    test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)  # 可以改成验证集
    with torch.no_grad():
        model = present_model
        if opt.cuda:
            model = model.cuda()
        model.eval()
        for iteration, batch in enumerate(test_data_loader, 1):
            input, target = batch[0], batch[1]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            out = model(input)
            out2 = out
            target2 = target
            psnr.update(calc_psnr(target2, out2), len(out2))
        print("--->This--epoch:{}--Avg--PSNR: {:.4f} dB".format(epoch_num, psnr.avg))
        wandb.log({'lr': optimizer.param_groups[0]["lr"], 'PSNR': psnr.avg})


def validate(present_model, optimizer, epoch_num):
    global max_avr_psnr

    print("----Start Validate----")
    psnr = AverageMeter()
    validate_set = get_validate_set()
    validate_data_loader = DataLoader(dataset=validate_set, batch_size=1, shuffle=True)
    with torch.no_grad():
        model = present_model
        if opt.cuda:
            model = model.cuda()
        model.eval()
        for iteration, batch in enumerate(validate_data_loader, 1):
            input, target = batch[0], batch[1]
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            out = model(input)
            out2 = out
            target2 = target
            psnr.update(calc_psnr(target2, out2), len(out2))
        print("---validate-->This-epoch:{}--Avg-PSNR: {:.4f} dB".format(epoch_num, psnr.avg))
        wandb.log({'lr': optimizer.param_groups[0]["lr"], 'PSNR': psnr.avg})

        max_avr_psnr = psnr.avg
        return max_avr_psnr


if __name__ == "__main__":
    main()
