'''
train.py

--type
spp
--activation
leaky
--batch-size
8
--epochs
20
--LR
0.00025
--optimize
adam
--save_interval
5
--img_size
608
--data
data/basket/basket.data
--rect
--lr_schedule
cosin
--expFolder
basket
--expID
test
'''

'''
detect.py
--source
/media/hkuit164/TOSHIBA/YOLOv3/data/basket/JPEGImages/
--view-img
'''

'''
convert pth to weights
python3 models.py
'''
