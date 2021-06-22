import os

vid_folder = '/media/hkuit164/WD20EJRX/1'

for name in os.listdir(vid_folder):
    path = os.path.join(vid_folder,name)
    cfg = '/media/hkuit164/WD20EJRX/result/best_weights/rgb/146/yolov3-original-1cls-leaky.cfg'
    weights = '/media/hkuit164/WD20EJRX/result/best_weights/rgb/146/best.weights'
    cmd = 'python detect.py --cfg {} --source {} --weights {}'.format(cfg,path,weights)
    os.system(cmd)