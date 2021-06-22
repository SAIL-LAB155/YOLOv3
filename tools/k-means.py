import numpy as np
import os
import cv2


def iou(box, clusters):
    # box: tuple or array, shifted to the origin
    # clusters: numpy array of shape(k,2), k is the num of clusters
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError('box has no area')
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算一个gt的bbox与k个anchors交并比均值
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """
    boxes: shape(r,2) gt的bbox, r是gt的bbox的个数
    k: anchors的个数
    dist:计算距离的方式
    """
    rows = boxes.shape[0]
    # 距离数组，每个gt的bbox对应k个anchor距离
    distances = np.empty((rows, k))
    # 上次gt的bbox距离最近的centroids的索引
    last_clusters = np.zeros((rows,))
    # 设置随机zhongzi
    np.random.seed()

    # 初始化k个质心
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 开始聚类
    while True:
        for row in range(rows):
            # 计算每个gt的bbox与k个anchor的距离，用1-IOU来计算
            distances[row] = 1 - iou(boxes[row], clusters)
        # 对每个bbox选择最近的anchor,记下索引
        nearest_clusters = np.argmin(distances, axis=1)
        if last_clusters.all() == nearest_clusters.all():
            break
        # 更新簇心
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters
    return clusters


if __name__ == "__main__":
    label_pth = '/media/hkuit164/TOSHIBA/YOLOv3/data/basket/txt'
    labels = sorted(os.listdir(label_pth))
    # imgs_pth = '/media/hkuit164/Sherry/sherry_data/swim_data/swim_gray'
    boxes = []
    for label in labels:
        with open(os.path.join(label_pth, label), 'r') as f:
            # img = cv2.imread(os.path.join(imgs_pth, label.replace(".txt", ".jpg")))
            # w, h, _ = img.shape
            w = 1920
            h = 1080
            lines = f.readlines()
            for line in lines:
                box = list(map(float, line.rstrip().lstrip().split()))[-2:]
                box[0] *= w
                box[1] *= h
                boxes.append(box)
    boxes = np.array(boxes).reshape(-1, 2)
    clusters = kmeans(boxes, 9)
    print(clusters)
