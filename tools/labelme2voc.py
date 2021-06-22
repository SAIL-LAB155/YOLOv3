import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
import tqdm
# # 1.标签路径
# labelme_path = "/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/rgb/txt/"  # 原始labelme标注数据路径
saved_path = "/media/hkuit164/WD20EJRX/voc_data/"  # 保存路径
# imgs_path = "/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/rgb/JPEGImages/"
# # 2.创建要求文件夹
# if not os.path.exists(saved_path + "Annotations"):
#     os.makedirs(saved_path + "Annotations")
# if not os.path.exists(saved_path + "JPEGImages/"):
#     os.makedirs(saved_path + "JPEGImages/")
# if not os.path.exists(saved_path + "ImageSets/Main/"):
#     os.makedirs(saved_path + "ImageSets/Main/")
#
# # 3.获取待处理文件
# files = glob(labelme_path + "*.txt")
# files = [i.split("/")[-1].split(".txt")[0] for i in files]
#
# # 4.读取标注信息并写入 xml
# for file in tqdm.tqdm(files):
#     txt_filename = labelme_path + file + ".txt"
#     txt_file = open(txt_filename, "r")
#     lines = txt_file.readlines()
#     height, width, channels = cv2.imread(imgs_path + file + ".jpg").shape
#     with codecs.open(saved_path + "Annotations/" + file + ".xml", "w", "utf-8") as xml:
#         xml.write('<annotation>\n')
#         xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
#         xml.write('\t<filename>' + file + ".jpg" + '</filename>\n')
#         xml.write('\t<source>\n')
#         xml.write('\t\t<database>The UAV autolanding</database>\n')
#         xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
#         xml.write('\t\t<image>flickr</image>\n')
#         xml.write('\t\t<flickrid>NULL</flickrid>\n')
#         xml.write('\t</source>\n')
#         xml.write('\t<owner>\n')
#         xml.write('\t\t<flickrid>NULL</flickrid>\n')
#         xml.write('\t\t<name>ChaojieZhu</name>\n')
#         xml.write('\t</owner>\n')
#         xml.write('\t<size>\n')
#         xml.write('\t\t<width>' + str(width) + '</width>\n')
#         xml.write('\t\t<height>' + str(height) + '</height>\n')
#         xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
#         xml.write('\t</size>\n')
#         xml.write('\t\t<segmented>0</segmented>\n')
#         for line in lines:
#             label, xc, yc, w, h = list(map(float, line.split()))
#             label = 'person'  # int(label)
#             xmin = int((xc - 0.5 * w) * width)
#             xmax = int((xc + 0.5 * w) * width)
#             ymin = int((yc - 0.5 * h) * height)
#             ymax = int((yc + 0.5 * h) * height)
#             if xmax <= xmin:
#                 pass
#             elif ymax <= ymin:
#                 pass
#             else:
#                 xml.write('\t<object>\n')
#                 xml.write('\t\t<name>' + label + '</name>\n')
#                 xml.write('\t\t<pose>Unspecified</pose>\n')
#                 xml.write('\t\t<truncated>1</truncated>\n')
#                 xml.write('\t\t<difficult>0</difficult>\n')
#                 xml.write('\t\t<bndbox>\n')
#                 xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
#                 xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
#                 xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
#                 xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
#                 xml.write('\t\t</bndbox>\n')
#                 xml.write('\t</object>\n')
#                 # print(txt_filename,xmin,ymin,xmax,ymax,label)
#         xml.write('</annotation>')
#
# # 5.复制图片到 VOC2007/JPEGImages/下
#
# image_files = glob(imgs_path + "*.jpg")
# print("copy image files to SWIMGRAY/JPEGImages/")
# for image in tqdm.tqdm(image_files):
#     shutil.copy(image, saved_path + "JPEGImages/")

# 6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')
total_files = glob("/media/hkuit164/WD20EJRX/voc_data/Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
# test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
# test
# for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
# split
train_files, val_files = train_test_split(total_files, test_size=0.15, random_state=42)
# train
for file in train_files:
    ftrain.write(file + "\n")
# val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
# ftest.close()