import os, sys, zipfile
import json


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + (box[2] - box[0]) / 2.0
    y = box[1] + (box[3] - box[1]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

json_file = '/home/hkuit164/Documents/tile_round1_train_20201231/train_annos.json'  # # Object Instance 类型的标注

datas = json.load(open(json_file, 'r'))

ana_txt_save_path = "../tianchi/train/"  # 保存的路径
if not os.path.exists(ana_txt_save_path):
    os.makedirs(ana_txt_save_path)

for data in datas:
    img_name = data['name']
    txt_name = img_name.split('.')[0]+'.txt'
    category = data['category']
    img_height , img_width = data['image_height'] ,data['image_width']
    box = convert((img_width, img_height),data['bbox'])
    with open(os.path.join(ana_txt_save_path,txt_name),'a') as f:
        f.write("%s %s %s %s %s\n" % (category, box[0], box[1], box[2], box[3]))

    # img_id = img["id"]
    # ana_txt_name = filename.split(".")[0] + ".txt"  # 对应的txt名字，与jpg一致
    # print(ana_txt_name)
    # f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
    # for ann in data['annotations']:
    #     if ann['image_id'] == img_id:
    #         # annotation.append(ann)
    #         # print(ann["category_id"], ann["bbox"])
    #         box = convert((img_width, img_height), ann["bbox"])
    #         if ann["category_id"] == 1:
    #             f_txt.write("%s %s %s %s %s\n" % (0, box[0], box[1], box[2], box[3]))
    # f_txt.close()