import pandas as pd
import csv
from utils.opt import opt
import os
import numpy as np

def write_test_csv(mp, mr, map, mf1):
    df = pd.read_csv(opt.csv_path)
    df_head = df[0:1]
    exist = os.path.exists('test_result.csv')
    with open('test_result.csv', 'a+') as f:
        f_csv = csv.writer(f)
        if not exist:
            title = list(df_head)[0:18]
            title.extend(['test_data', 'P', 'R', 'mAP', 'F1'])
            f_csv.writerow(title)
        info_list = np.array(df.loc[df['ID'] == opt.id]).tolist()[0][0:18]
        info_list.extend([opt.data.split('/')[-1], mp, mr, map, mf1])
        f_csv.writerow(info_list)

def error_ana(all_path, correct_img):
    exist = os.path.exists('error_img.csv')
    with open('error_img.csv', 'a+') as f:
        f_csv = csv.writer(f)
        if not exist:
            f_csv.writerow(all_path)
        f_csv.writerow(correct_img)