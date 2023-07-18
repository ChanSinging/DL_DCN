import os
path = './datasets/train/gt/'      # 输入文件夹地址
files = os.listdir(path)   # 读入文件夹
num_file = len(files)       # 统计文件夹中的文件个数
print(num_file)             # 打印文件个数
sum = 0
for i in range(num_file):
    path2 = path+'/{:0>3d}/'.format(i)  # gt/001/
    print(path2)
    path2 = os.listdir(path2)
    num_video = len(path2)
    sum = sum + num_video
    print('next!!!')
# 输出所有文件名
print(sum)
print("所有文件名:")


# 总共有57723个图片 57723
