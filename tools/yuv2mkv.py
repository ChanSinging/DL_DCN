import os
from os import path

command = 'ffmpeg -pix_fmt yuv420p -s 960x536 -i 001_960x536_218.yuv output.mkv'

source = path.normpath(r'raw/')
videoList = os.listdir(source)

for name in videoList:
    # 001_960x536_218.yuv
    fileName = name[0:-4]
    cmd = "/home/lixiaodong/CBRENA/tool/ffmpeg -pix_fmt yuv420p -s 960x536 -i {} -c:v libx265 {}.mkv".format(name, fileName)
    print(cmd)
    os.system(cmd)
