import os
from os import path

command = 'ffmpeg -pix_fmt yuv420p -s 960x536 -i 001_960x536_218.yuv output.mkv'

source = path.normpath(r'/home/pxk/CBREN/datasets_test/')
videoList = os.listdir(source)
print(videoList)

for name in videoList:
    # 001_960x536_218.yuv
    # fileName = name[0:-4]
    raw_video_name = os.path.basename(name).split(".")[0]
    _res = raw_video_name.split("_")[1]
    width = _res.split("x")[0]
    height = _res.split("x")[1]
    # nfs = raw_video_name.split("_")[2]
    cmd1 = "ffmpeg -pix_fmt yuv420p -s {}x{} -r 30 -i {}.yuv -c:v libx265 -b:v 300k -x265-params pass=1:log-level=error -f null /dev/null".format(width, height, raw_video_name)
    cmd2 = "ffmpeg -pix_fmt yuv420p -s {}x{} -r 30 -i {}.yuv -c:v libx265 -b:v 300k -x265-params pass=2:log-level=error {}.mkv".format(width, height, raw_video_name, raw_video_name)
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)