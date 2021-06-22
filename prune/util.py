import time
import torch
from models import *
import cv2

def obtain_avg_forward_time(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time


if __name__ == '__main__':
    cfg = '/media/hkuit164/WD20EJRX/result/best_finetune/black/SLIM-prune_0.93_keep_0.1/prune_0.93_keep_0.1.cfg'
    img_size = 416
    weights = '/media/hkuit164/WD20EJRX/result/best_finetune/black/SLIM-prune_0.93_keep_0.1/best.weight'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'
    print(device)
    model = Darknet(cfg, (img_size, img_size)).to(device)
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights)['model'])
    else:
        load_darknet_weights(model, weights)
    model.eval()
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)
    time = obtain_avg_forward_time(random_input,model)
    print(time)