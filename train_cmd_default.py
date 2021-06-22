#-*-coding:utf-8-*-

cmds = [
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb_adam.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0002 --expFolder rgb_adam  --expID 2',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb_adam.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0003 --expFolder rgb_adam  --expID 3',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb_adam.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0004 --expFolder rgb_adam  --expID 4',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb_adam.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0002 --expFolder test  --expID 2_sgd',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb_adam.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0003 --expFolder rgb_adam  --expID 3_sgd',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb_adam.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0004 --expFolder rgb_adam  --expID 4_sgd',

#
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0001 --expFolder rgb	--expID 1',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0002 --expFolder rgb	--expID 2',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0003 --expFolder rgb	--expID 3',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0004 --expFolder rgb	--expID 4',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0005 --expFolder rgb	--expID 5',
# 'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0006 --expFolder rgb	--expID 6',

# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0001 --expFolder rgb	--expID 11',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0002 --expFolder rgb	--expID 12',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0003 --expFolder rgb	--expID 13',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0004 --expFolder rgb	--expID 14',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize adam --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0005 --expFolder rgb	--expID 15',
# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0006 --expFolder rgb	--expID 16',


# 'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0003 --expFolder rgb	--expID 3_sgd',
# 'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0005 --expFolder rgb	--expID 5_sgd',
# 'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0003 --expFolder rgb	--expID 13_sgd',
# 'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0005 --expFolder rgb	--expID 15_sgd'

# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0007 --expFolder rgb	--expID 7',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0007 --expFolder rgb	--expID 17',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0008 --expFolder rgb	--expID 8',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0008 --expFolder rgb	--expID 18',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0009 --expFolder rgb	--expID 9',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 1 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0009 --expFolder rgb	--expID 19',
#
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0001 --expFolder rgb_normal	--expID 1',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0002 --expFolder rgb_normal	--expID 2',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0003 --expFolder rgb_normal	--expID 3',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0004 --expFolder rgb_normal	--expID 4',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0005 --expFolder rgb_normal	--expID 5',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0006 --expFolder rgb_normal	--expID 6',
#
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0001 --expFolder rgb_normal	--expID 11',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0002 --expFolder rgb_normal	--expID 12',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0003 --expFolder rgb_normal	--expID 13',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0004 --expFolder rgb_normal	--expID 14',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0005 --expFolder rgb_normal	--expID 15',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0006 --expFolder rgb_normal	--expID 16',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0008 --expFolder rgb_normal	--expID 18',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0009 --expFolder rgb_normal	--expID 9',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0009 --expFolder rgb_normal	--expID 19',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0007 --expFolder rgb_normal	--expID 7',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule step --s 0.0007 --expFolder rgb_normal	--expID 17',
# 'CUDA_VISIBLE_DEVICES=0 python train.py --type original --activation leaky --batch-size 16 --freeze False --prune 0 --epochs 100 --LR 0.00025 --optimize sgd --weights weights/rgb.weights --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/rgb/rgb.data --sr True --lr_schedule cosin --s 0.0008 --expFolder rgb_normal	--expID 8',



# 'CUDA_VISIBLE_DEVICES=1 python train.py --type original --activation leaky --batch-size 16 --freeze False  --weights weights/darknet53.conv.74 --epochs 200 --LR 0.0025 --optimize adam  --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/COCO/coco.data --lr_schedule cosin  --expFolder COCO	--expID 10lr_noapex',
'CUDA_VISIBLE_DEVICES=2 python train.py --type original --activation leaky --batch-size 16 --freeze False  --weights weights/darknet53.conv.74 --epochs 200 --LR 0.0025 --optimize adam  --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/COCO/coco.data --lr_schedule step  --expFolder COCO	--expID 10lr_noapex_step'

]

def check_name(cmd):
    if '--multi-scale True' in cmd:
        cmd=cmd.replace('--multi-scale True','--multi-scale')
    if '--multi-scale False' in cmd:
        cmd = cmd.replace('--multi-scale False', '')
    if '--rect True' in cmd:
        cmd=cmd.replace('--rect True','--rect')
    if '--rect False' in cmd:
        cmd=cmd.replace('--rect False','')
    if '--freeze True' in cmd:
        cmd=cmd.replace('--freeze True','--freeze')
    if '--freeze False' in cmd:
        cmd=cmd.replace('--freeze False','')
    if '--sr True' in cmd:
        cmd = cmd.replace('--sr True', '--sr')
    if '--sr False' in cmd:
        cmd = cmd.replace('--sr False', '')
    return cmd

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    cmd=check_name(cmd)
    print(cmd)
    os.system(cmd)
