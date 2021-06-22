#-*-coding:utf-8-*-

import os
first_floder = '/media/hkuit164/WD20EJRX/416black'
for inside_floder in os.listdir(first_floder):
     for name in os.listdir(first_floder+'/'+inside_floder):
         # print(name)
         if name.endswith('.weights'):
             print(inside_floder+'_'+name)
             weight_name = first_floder + '/' + inside_floder + '/' + name
             cmd = 'python test.py --cfg cfg/yolov3-spp-1cls-leaky.cfg --data data/black_test.data ' \
                  '--weights {} --batch-size 16 --img-size 416 ' \
                  '--conf-thres 0.1'.format(weight_name)
             os.system(cmd)
