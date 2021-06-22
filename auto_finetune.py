import os
import shutil
from prune.config import data, batch_size, epoch, ms
from prune.config import finetune_folders as folders

cmds = []
for folder in folders:
    cfg, model = "", ""
    path_ls = folder.split("/")
    wdir = path_ls[1] + "/" + path_ls[2]

    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    for file in files:
        if "cfg" in file:
            cfg = file
        elif ".pt" in file or ".weight" in file:
            model = file
        else:
            continue
    # wdir = os.path.join(wdir_tmp,model.split('/')[-2])
    # shutil.copy(cfg, os.path.join("finetune", wdir))
    assert cfg != "" and model != "", "Missing file in {}! (cfg or weight missed)".format(folder)
    cmds.append("CUDA_VISIBLE_DEVICES=3 python train.py --finetune --optimize adam --expFolder {} --cfg {} --weights {} --data {} --epochs {} --batch-size {} "
                "--multi-scale {}".format(wdir, cfg, model, data, epoch, batch_size, ms))

for cmd in cmds:
    cmd = cmd.replace("--multi-scale False", "")
    cmd = cmd.replace("--multi-scale True", "--multi-scale")
    # print(cmd)
    os.system(cmd)
    # print(cmd)
