import argparse

parser = argparse.ArgumentParser(prog='train.py')

'-----------------------------------training-----------------------------------------'

parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
parser.add_argument('--cfg', type=str, default='cfg/yolov3-original-1cls-leaky.cfg', help='cfg file path')
parser.add_argument('--t_cfg', type=str, default='', help='teacher model cfg file path for knowledge distillation')
parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
parser.add_argument('--test_data', type=str, default='data/ceiling.data', help='*.data file path')
parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
parser.add_argument('--weights', type=str, default='', help='initial weights')  # i.e. weights/darknet.53.conv.74
parser.add_argument('--t_weights', type=str, default='', help='teacher model weights')
parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--sr', action='store_true',help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.001, help='scale sparse rate')
parser.add_argument('--prune', type=int, default=1, help='0:nomal prune 1:other prune ')
parser.add_argument('--freeze', action='store_true', help='freeze layers ')
parser.add_argument('--expID', type=str, default='0', help='model number')
parser.add_argument('--LR', type=float, default=0.001, help='learning rate')
parser.add_argument('--type', type=str, default='spp', help='yolo type(spp,normal,tiny)')
parser.add_argument('--activation', type=str, default='leaky', help='activation function(leaky,swish,mish)')
parser.add_argument('--expFolder', type=str, default='gray', help='expFloder')
parser.add_argument('--save_interval', default=10, type=int, help='interval')
parser.add_argument('--optimize', type=str, default='sgd', help='optimizer(adam,sgd)')
parser.add_argument('--lr_schedule', type=str, default='cosin', help='lr_schedule: step or cosin')

'--------------------------------------testing------------------------------------'

parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument('--id', type=int, default=000, help='inference size (pixels)')
parser.add_argument('--csv_path', type=str, default='', help='path to weights file')
parser.add_argument('--write_csv', action='store_true', help='save test csv file')
parser.add_argument('--finetune', action='store_true', help='finetune')
'----------------------------------slim-prune-------------------------------------'
parser.add_argument('--global_percent', type=float, default=0.8, help='global channel prune percent')
parser.add_argument('--layer_keep', type=float, default=0.01, help='channel keep percent per layer')
parser.add_argument('--only_metric', type=bool, default=False, help="whether save cfg and model")
'----------------------------------all-prune-------------------------------------'
parser.add_argument('--shortcuts', type=int, default=8, help='how many shortcut layers will be pruned,\
    pruning one shortcut will also prune two CBL,yolov3 has 23 shortcuts')
opt = parser.parse_args()

print(opt)