# import numpy as np;
# import cv2

# img = cv2.imread('../dataset/cbren_010.png',1)
# cutimg = img[250:280,110:130]
# print(cutimg)
# # cv2.imshow('origin',img)
# # cv2.imshow('image',cutimg);
# cv2.imwrite('{}cut.jpg'.format(path[11:20]), cutimg)
# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()
# B:1920x1280  640
# D:416x240
# C:832x480
# E:1280x720
from PIL import Image


path = '/home/zhangqianyu/qb27/DL_DCN 030.png'

img = Image.open(path)

print(img.size)
#cropped = img.crop((25, 650, 175, 800))  # (left, upper, right, lower) 左上，右下    people009
#cropped = img.crop((300, 260, 400, 360))  #basketball
#cropped = img.crop((450, 450, 550, 550))   #park
#cropped = img.crop((1675, 225, 1775, 325))  #traffic
cropped = img.crop((1250, 200, 1350, 300))  #cauits

Image._show(cropped)
cropped.save("{}.png".format(path[23:-4]))

