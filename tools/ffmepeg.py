import os
from os import path

# 批量转化mp4格式
# for %a in ("X:\xxx\xx\*.mkv") do ffmpeg -i "%a" -c:v copy -c:a aac "Y:\yyy\yy\%~na.mp4"

# 获取文件路径，获取文件名称列表
source = path.normpath(r'/home/chenxingying/STDF-PyTorch/dataset/ntire/training_raw/')
videoList = os.listdir(source)

# 只选择目录下的mkv文件
for Sname in videoList:
    if not Sname.endswith("mkv"):
        videoList.remove(Sname)

# # 执行ffmpeg命令
# for i in videoList:
#     output = i[0:-4]
#     cmd = "ffmpeg -i /home/chenxingying/STDF-PyTorch/dataset/ntire/training_raw/%s -c:v copy -c:a aac /home/chenxingying/code/CBREN/train_mp4/%s.mp4" %(i, output)
#     os.system(cmd)

# 执行ffmpeg命令 ffmpeg -i xxx.mkv -pix_fmt yuv420p xxx.yuv
for i in videoList:
    output = i[0:-4]
    cmd = "ffmpeg -i /home/chenxingying/STDF-PyTorch/dataset/ntire/training_raw/%s -pix_fmt yuv420p /home/chenxingying/STDF-PyTorch/dataset/ntire/training_yuv/%s.yuv" %(i, output)
    os.system(cmd)
