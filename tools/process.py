'''
allocate train and validation images for training
'''
import glob, os

# current_dir = os.path.dirname(os.path.abspath(__file__))

current_dir= '/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/data/yoga/JPEGImages'
# print(current_dir)
# percentage_test = 10
percentage_test = 10
file_train = open(current_dir[:-10]+'train.txt', 'w')
file_test = open(current_dir[:-10]+'val.txt', 'w')

counter = 1
index_test = round(100 / percentage_test)
# print(list(glob.iglob(os.path.join(current_dir, "*.jpg"))))
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        file_test.write('./data/yoga/JPEGImages' + "/" + title + ".jpg" + "\n")
    else:
        file_train.write('./data/yoga/JPEGImages' + "/" + title + ".jpg" + "\n")
        counter = counter + 1
