import os,shutil
from tqdm import tqdm
sourceDir='/home/user/Documents/yolov3-channel-and-layer-pruning/data/COCO/COCO/train/'
targetDir='/home/user/Documents/yolov3-channel-and-layer-pruning/data/COCO/txt/'
for root, dirs, files in os.walk(sourceDir):
    for file in tqdm(files,desc="Waiting..."):
        shutil.copy(os.path.join(root,file),targetDir)