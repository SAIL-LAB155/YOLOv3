import onnxruntime as ort
import cv2
import numpy as np
import torch
import os

def image_normalize(img_name, size=224):
    image_normalize_mean = [0.485, 0.456, 0.406]
    image_normalize_std = [0.229, 0.224, 0.225]
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    image_array = np.float32(image_array)
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

ls = []
name = os.listdir('/media/hkuit164/WD20EJRX/CNN_classification/test/test_img')
for img_name in name:
    image_path = '/media/hkuit164/WD20EJRX/CNN_classification/test/test_img/' + img_name
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_tensor_list = []
    img_array = image_normalize(img_raw,size=224)
    ort_session = ort.InferenceSession('/media/hkuit164/WD20EJRX/CNN_classification/model.onnx')
    output = ort_session.run(None, {'input.1': img_array})
    ls.append(output)
for (i,j) in zip(ls,name):
    print(j)
    print(i)
