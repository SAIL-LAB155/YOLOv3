models = {
    "sparse_result/0908/test1/backup170.pt": "cfg/yolov3-1cls.cfg",
    "sparse_result/0908/test1/best.pt": "cfg/yolov3-1cls.cfg",
}

data = "data/gray/gray.data"

# Sparse option
sparse_type = ["shortcut", "ordinary"]
p_max, p_min = 99, 50

# Prune option
only_metric = False

prune = [0.85, 0.9, 0.92, 0.95]
shortcut_p = [0.85, 0.9, 0.92, 0.95]
layer_num = [8, 12]
slim_params = [(0.93, 0.1)]
all_prune_params = [(10, 0.95, 0.01), (15, 0.96, 0.01)]

# Finetune option
finetune_folders = [
    "prune_result/0909_test/test1-backup170-all_prune-prune_0.95_keep_0.01_10_shortcut",
    "prune_result/0909_test/test1-backup170-layer_prune-prune_12_shortcut",
    "prune_result/0909_test/test1-backup170-ordinary_prune-prune_0.85",
    "prune_result/0909_test/test1-backup170-shortcut_prune-prune_0.9",
]
batch_size = 4
epoch = 100
ms = True

# (Distillation)
t_cfg = "weights/teacher/146/yolov3-original-1cls-leaky.cfg"
t_weight = "weights/teacher/146/best.pt"
