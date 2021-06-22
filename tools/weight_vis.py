from models import *
import torch
from utils.utils import *
import numpy as np


cfg_path = '/media/hkuit164/TOSHIBA/YOLOv3/cfg/yolov3-original-1cls-leaky.cfg'

img_size = (416,416)
weight = '/media/hkuit164/TOSHIBA/YOLOv3/weights/rgb/1/backup90.pt'
model = Darknet(cfg=cfg_path,img_size=img_size)
device = torch_utils.select_device(device='cpu')

if weight.endswith('.pt'):
    model.load_state_dict(torch.load(weight, map_location=device)['model'])
else:
    print('Wrong format')

model.to(device).eval()
all_weight = np.zeros((1,1))
for i in list(model.named_parameters()):
    if 'BatchNorm2d.weight' in i[0]:
        w_flatten = np.array(i[1].T.reshape(-1, 1).detach())
        all_weight = np.concatenate([all_weight,w_flatten], axis=0)
plt.hist(all_weight, bins=500, color="b", density=True, range=[-0.0005, 0.0005])
        # plt.title("epoch=" + str(i) + " loss=%.2f ,acc=%.3f" % (score[0], score[1]))
        # plt.savefig("mnist_model_weights_hist_%d.png" % (i))
plt.show()

# Conv2d