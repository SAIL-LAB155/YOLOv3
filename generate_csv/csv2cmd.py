import csv    #加载csv包便于读取csv文件
import os
from generate_csv.config import task_folder, batch_folder ,include_cuda


class Generate_csv():
    def __init__(self, csv_name):
        self.csv_name = csv_name

    def change_name(self, name):
        if name == "TRUE":
            return 'True'
        elif name == "FALSE":
            return "False"
        else:
            return name

    def step_1(self,):
        out_name = self.csv_name[:-4] + ".txt"
        with open(self.csv_name,'r') as f:
            csv_reader_lines = csv.reader(f)
            data = [line for line in csv_reader_lines]
            opt = [item for item in data[0]]

        if include_cuda:
            begin = "'CUDA_VISIBLE_DEVICES= python train.py "
        else:
            begin = "'python train.py "
        cmds = []
        for idx, mdl in enumerate(data[1:]):
            tmp = ""
            valid = False
            for o, m in zip(opt, mdl):
                if m != "":
                    tmp += "--"
                    tmp += o
                    tmp += " "
                    tmp += self.change_name(m)
                    tmp += " "
                    valid = True

            tmp += "--expFolder {}".format(batch_folder)
            tmp += "\t--expID {}".format(idx + 1)
            cmd = begin + tmp + "'\n"
            if valid:
                cmds.append(cmd)
            else:
                cmds.append("\n")

        with open(out_name, "a+") as out:
            for c in cmds:
                out.write(c)

    def step_2(self, cuda):
        with open("{}.txt".format(os.path.join(task_folder, batch_folder, batch_folder)), "r") as f:
            lines = [line for line in f.readlines() if line != "\n"]

        train_begin, train_end = 24, 24
        CUDA = cuda
        target_cmds = lines[:]

        # if CUDA != -1:
        #     cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in target_cmds]
        # else:
        #     cmds = [cmd[0] + cmd[23:-1] + ",\n" for cmd in target_cmds]
        cmds = [cmd[:22] + cmd[22:-1] + ",\n" for cmd in target_cmds]
        with open("{}/cmds.txt".format(os.path.join(task_folder, batch_folder)), "a+") as cf:
            for cmd in cmds:
                cf.write(cmd)
            cf.write("\n")


if __name__ == '__main__':
    csv_name = "{}.csv".format(os.path.join(task_folder, batch_folder, batch_folder))
    gener_csv = Generate_csv(csv_name)
    gener_csv.step_1()
    cuda_num = 1
    gener_csv.step_2(cuda_num)

