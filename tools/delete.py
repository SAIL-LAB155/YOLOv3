'''
delete txt file if it strats with 1
'''
import os
import re
name = os.listdir('/home/user/Documents/yolov3-channel-and-layer-pruning/data/rgb/txt')
for i in range(len(name)):
    path = os.path.join('/home/user/Documents/yolov3-channel-and-layer-pruning/data/rgb/txt',name[i])
    print(path)
    f = open(path, 'r')
    alllines = f.readlines()
    f.close()
    f = open(path, 'w+')
    for eachline in alllines:
        a = re.sub('1 ', '0 ', eachline)
        f.writelines(a)
    f.close()