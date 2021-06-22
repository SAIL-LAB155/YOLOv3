import torch

from models import ONNX_EXPORT, Darknet, load_darknet_weights


def convert_onnx(img_size, model):

    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True,opset_version=11)
        return

def load_model(cfg, img_size, weights):
    model = Darknet(cfg, img_size)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cuda:0')['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    return model

if __name__ == '__main__':
    img_size = (416,416)
    cfg = '/media/hkuit164/TOSHIBA/YOLOv3/cfg/yolov3-original-1cls-leaky.cfg'
    weights= '/media/hkuit164/TOSHIBA/YOLOv3/weights/best.weight'
    model = load_model(cfg, img_size, weights)
    model.to('cpu').eval()
    convert_onnx(img_size, model)