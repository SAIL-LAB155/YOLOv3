#-*-coding:utf-8-*-
patience = 9
warm_up = 6
patience_decay = {1:0.8,2:0.5,3:0}
device = "cuda:0"
computer = 'sean'
epoch = 20
convert_weight = True
weight_dest = "grayE-4"
sparse_param = []
# Hyperparameters (j-series, 50.5 mAP yolov3-320) -d by @ktian08 https://github.com/ultralytics/yolov3/issues/310
hyp = {'giou': 1.0,  # 1.582 giou loss gain
       'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 31.35,  # 21.35obj loss gain (*=80 for uBCE with 80 classes)
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou training threshold
       'lr0': 0.001324,  # 0.002324 initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0004569,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.10,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)

sparse_decay = {0.1: 0.1, 0.01: 0.01}

