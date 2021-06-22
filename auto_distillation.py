from prune.config import data, batch_size, epoch, t_cfg, t_weight, ms
from prune.config import finetune_folders as folders
import shutil

import os

cmds = []
for folder in folders:
    cfg, model = "", ""
    path_ls = folder.split("/")
    wdir = path_ls[1] + "-" + path_ls[2]

    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    for file in files:
        if "cfg" in file:
            cfg = file
        elif ".pt" in file or ".weight" in file:
            model = file
        else:
            continue

    assert cfg != "" and model != "", "Missing file in {}! (cfg or weight missed)".format(folder)
    cmds.append("python train.py --wdir finetune/{}_distilled --cfg {} --weights {} --data {} --epochs {} "
                "--batch-size {} --multi-scale {} --t_cfg {} --t_weight {}".format(wdir, cfg, model, data, epoch,
                                                                                   batch_size, ms, t_cfg, t_weight))

for cmd in cmds:
    cmd = cmd.replace("--multi-scale False", "")
    cmd = cmd.replace("--multi-scale True", "--multi-scale")
    os.system(cmd)
    # print(cmd)
