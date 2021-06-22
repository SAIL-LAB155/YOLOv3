import cv2
import os

img_path = '/home/user/Documents/yolov3-channel-and-layer-pruning/data/yoga/JPEGImages'
for name in os.listdir(img_path):
    img = cv2.imread(os.path.join(img_path,name))
    cv2.imshow("img",img)
    print(name)
    cv2.waitKey(1)
    # if name.endswith('png'):
    #     print("______________________________________________________________________________________________________")