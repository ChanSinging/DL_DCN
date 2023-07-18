f = open("../rootmodel/DL-DCN-edvr.txt", "r")  # 设置文件对象
info = f.read()
infolist = info.split('\n')
print(infolist[2])
