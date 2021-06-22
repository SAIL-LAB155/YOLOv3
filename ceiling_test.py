import traceback

import pandas as pd

from models import *

path = '/media/hkuit164/MB155_3/result/ceiling/ceiling_result_sean.csv'
weight_folder='/media/hkuit164/MB155_4/ceiling'
data_folder = '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/test/'
df = pd.read_csv(path)
# name is id
for name in os.listdir(weight_folder):
    try:
        type = df[df['ID'] == int(name)][:]['tpye']
        activate = df[df['ID'] == int(name)][:]['activation']
        img_size = df[df['ID'] == int(name)][:]['img_size']
        # data = df[df['ID'] == int(name)][:]['data']
        weight_path = os.path.join(weight_folder,name)
        cfg = os.path.join('cfg','yolov3-'+list(type)[0]+'-1cls-'+list(activate)[0]+'.cfg')
        result=[]
        weight_name = os.path.join(weight_path, 'best.weight')
        if not os.path.exists(os.path.join(weight_path,'best.weight')):
            if not os.path.exists(os.path.join(weight_path,'best.pt')):
                print('Not found best.pt')
            else:
                convert(cfg=cfg, weights=os.path.join(weight_path,'best.pt'))
                #for data
                data = '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/test/ceiling/ceiling.data'
                cmd = 'python test.py --cfg {0} --data {1} ' \
                      '--weights {2} --batch-size 8 --img-size {3} ' \
                      '--conf-thres 0.5 --id {4} --csv_path {5} --write_csv'.format(cfg, data, weight_name,list(img_size)[0], int(name),path)
                os.system(cmd)
        else:

            data = '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/test/ceiling/ceiling.data'
            cmd = 'python test.py --cfg {0} --data {1} ' \
              '--weights {2} --batch-size 8 --img-size {3} ' \
              '--conf-thres 0.5 --id {4} --csv_path {5} --write_csv'.format(cfg, data, weight_name,list(img_size)[0], int(name), path)
            # print(cmd)
            os.system(cmd)
    except:
        if os.path.exists('test_error.txt'):
            os.remove('test_error.txt')
        with open('test_error.txt', 'a+') as f:
            f.write(name)
            f.write('\n')
            f.write('----------------------------------------------\n')
            traceback.print_exc(file=f)