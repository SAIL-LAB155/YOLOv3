# -*- coding: utf-8 -*-
import os
img_path = '/home/user/Documents/yolov3-channel-and-layer-pruning/caizhen/data'
anno_path = '/home/user/Documents/yolov3-channel-and-layer-pruning/caizhen/txt'

def contrastDir(img_path,anno_path):
    jpg_list = []
    xml_list = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                jpg_list.append(os.path.splitext(file)[0])
    for root, dirs, files in os.walk(anno_path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                xml_list.append(os.path.splitext(file)[0])

    diff = set(jpg_list).difference(set(xml_list))
    diff2 = set(xml_list).difference(set(jpg_list))
    print(diff)
    # for name in diff:
    #     print("No corresponding XML file", name + ".jpg")
    #
    #     os.remove(anno_path+'\\'+ name+'.txt')
        # os.remove(img_path + '\\' + name + '.jpg')

if __name__ == '__main__':

    contrastDir(img_path,anno_path)