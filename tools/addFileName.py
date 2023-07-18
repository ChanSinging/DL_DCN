import os
from os import path
import os.path as op

# 获取文件路径，获取文件名称列表
source = path.normpath(r'../../../dataset/MFQEv2_dataset/ntire/')
videoList = os.listdir(source)
videoList = sorted(videoList)
# print(videoList[0][0:3])

f = open("info.txt", "r")   #设置文件对象
info = f.read()
infolist = info.split( )  # spilr \t and \n
# print(infolist)

video_start = 0
frame_start = 1  # add 3 per time
video_file = 0
video_path = "../../../dataset/MFQEv2_dataset/ntire/"
video_size = '960x536'

def change_frames():
    for i in range(201):
        # print(video_start)
        if video_start == 398:
            break
        video_num = infolist[video_start]
        frame_num = infolist[frame_start]
        print('{} == {}'.format(video_num, videoList[video_file][0:3]))
        print(videoList[video_file][0:3])
        if videoList[video_file][0:3] == video_num:
            print("convert successfully")
            src = os.path.join(video_path, video_num + '.yuv')
            dct = os.path.join(video_path, video_num + '_{}'.format(frame_num)+'.yuv')  # target
            print(dct)
            os.rename(src, dct)
        video_start = video_start + 2
        video_file = video_file + 1
        frame_start = frame_start + 2
        # if video_num == 200:
        #     break


def change_size():
    video_length = len(videoList)
    print(video_length)
    for name in videoList:
        raw_video_name = os.path.basename(name).split(".")[0]
        video_name = raw_video_name.split("_")[0]
        # print(_res)
        src = os.path.join(video_path, raw_video_name + '.mkv')
        # print(src)
        dct = os.path.join(video_path, video_name + '_' + video_size + '_' + raw_video_name.split("_")[1] + '.mkv')
        os.rename(src, dct)
        print('convert successfully!')


if __name__ == '__main__':
    change_size()