import torch


def fix_parameter(model):
    # f = open("rootmodel/DL-DCN-edvr.txt", "r")  # 设置文件对象
    # info = f.read()
    # infolist = info.split('\n')
    # i = 0
    for k in model.named_parameters():
        # if k == infolist[i]:
        k.requires_grad = False
        print(k)

    return model

if __name__ == "__main__":
    # model = CVQnet()
    model_dict = model.state_dict()
    pretrained_dict = torch.load('./checkpoint/CVQENet.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)