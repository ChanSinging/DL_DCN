import os

path = "/home/lixiaodong/CBRENA/CBREN/datasets/validate/ntire_cbr/KristenAndSara_1280x720_60/"
path2 = 'D/'

for file in os.listdir(path2):
    num = file.split(".")[0].split("_")[-1]
    print('{}'.format(num))
    filename_change = num.zfill(0) + ".png"
    os.rename(os.path.join(path2, file), os.path.join(path2, filename_change))
