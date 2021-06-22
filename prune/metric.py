from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse
from utils.compute_flops import print_model_param_flops, print_model_param_nums


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output


    wdir, model_name = opt.weights.split("/")[0], opt.weights.split("/")[1]
    file_name = os.path.join(wdir, "metric.txt")

    if not os.path.exists(file_name):
        file = open(file_name, "w")
        file.write(('\n' + '%70s' * 1) % "Model")
        file.write(('%15s' * 1) % "FLOPs")
        file.write(('%10s' * 9) % ("Time", "Params", "P", "R", "mAP", "F1", "test_GIoU", "test_obj", "test_cls\n"))
    else:
        file = open(file_name, "a+")

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ', opt.weights)

    eval_model = lambda model:test(model=model,cfg=opt.cfg, data=opt.data, batch_size=16, img_size=img_size, conf_thres=0.1)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nTesting model {}:".format(model_name))
    with torch.no_grad():
        metric, _ = eval_model(model)
    print(metric)
    metric = [round(m, 4) for m in metric]

    flops = print_model_param_flops(model)
    params = print_model_param_nums(model)

    random_input = torch.rand((1, 3, img_size, img_size)).to(device)
    forward_time, _ = obtain_avg_forward_time(random_input, model)
    forward_time = round(forward_time, 4)

    file.write(('\n' + '%70s' * 1) % ("{}".format(model_name)))
    file.write(('%15s' * 1) % ("{}".format(flops)))
    file.write(('%10s' * 9) % ("{}".format(forward_time), "{}".format(params), "{}".format(metric[0]), "{}".format(metric[1]),
                               "{}".format(metric[2]), "{}".format(metric[3]), "{}".format(metric[4]),
                               "{}".format(metric[5]), "{}\n".format(metric[5]),))
