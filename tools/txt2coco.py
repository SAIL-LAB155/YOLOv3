import os
import json
import numpy as np
import shutil
import cv2
from sklearn.model_selection import train_test_split
import tqdm

# 0为背景
classname_to_id = {"person": 1}


class Yolo2Coco:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_pth):
        json.dump(instance, open(save_pth, 'w', encoding='utf-8'))

    def to_coco(self, label_ls, label_pth, imgs_pth):
        self._init_categories()
        for label in tqdm.tqdm(label_ls):
            img_pth = os.path.join(imgs_pth, label.replace('.txt', '.jpg'))
            img = cv2.imread(img_pth)
            height, width = img.shape[0], img.shape[1]
            lines = self.read_txtfile(label_pth, label)
            self.images.append(self._image(img, img_pth))
            for line in lines:
                cl, xc, yc, w, h = list(map(float, line.strip().split()))
                cl = int(cl)
                xc *= width
                w *= width
                yc *= height
                h *= height
                annotation = self._annotations(cl, xc, yc, w, h)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # init categories
    def _init_categories(self):
        """
        'categories': [{'id': 0, 'name': person}, ...]
        """
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _image(self, img, path):
        """
        'images': [
        {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
        },
        ...
        ],
        """
        image = {}
        h, w = img.shape[0], img.shape[1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    def _annotations(self, cl, xc, yc, w, h):
        """
        'annotations': [
        {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
        },
        ...
        ],
        """
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = 1
        annotation['segmentation'] = [[]]
        annotation['bbox'] = self._get_box(xc, yc, w, h)
        annotation['iscrowd'] = 0
        annotation['area'] = w * h
        return annotation

    def read_txtfile(self, label_pth, label):
        with open(os.path.join(label_pth, label), 'r') as f:
            lines = f.readlines()
        return lines

    def _get_box(self, xc, yc, w, h):
        x1 = xc - 0.5 * w
        y1 = yc - 0.5 * h
        return x1, y1, w, h


if __name__ == "__main__":
    label_pth = "/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/yoga/1"
    imgs_pth = "/media/hkuit164/TOSHIBA/nanodet/data/yoga"
    saved_coco_pth = "/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/yoga/2/"
    # make dir for coco
    if not os.path.exists("%syoga_coco/annotations" % saved_coco_pth):
        os.makedirs("%syoga_coco/annotations/" % saved_coco_pth)
    if not os.path.exists("%syoga_coco/images/train2017/" % saved_coco_pth):
        os.makedirs("%syoga_coco/images/train2017" % saved_coco_pth)
    if not os.path.exists("%syoga_coco/images/val2017/" % saved_coco_pth):
        os.makedirs("%syoga_coco/images/val2017" % saved_coco_pth)

    # get all the labels
    label_ls = os.listdir(label_pth)
    # imgs_ls = glob.glob(imgs_pth + '/*.jpg')

    # split train and val
    train_ls, valid_ls = train_test_split(label_ls, test_size=0.12)

    # convert2coco for train set
    y2c_train = Yolo2Coco()
    train_instance = y2c_train.to_coco(train_ls, label_pth, imgs_pth)
    y2c_train.save_coco_json(train_instance, '%syoga_coco/annotations/instances_train2017.json' % saved_coco_pth)
    for file in train_ls:
        shutil.copy(os.path.join(imgs_pth, file.replace("txt", "jpg")),
                    "%syoga_coco/images/train2017/" % saved_coco_pth)
    for file in valid_ls:
        shutil.copy(os.path.join(imgs_pth, file.replace("txt", "jpg")), "%syoga_coco/images/val2017/" % saved_coco_pth)

    y2c_val = Yolo2Coco()
    val_instance = y2c_val.to_coco(valid_ls, label_pth, imgs_pth)
    y2c_val.save_coco_json(val_instance, '%syoga_coco/annotations/instances_val2017.json' % saved_coco_pth)