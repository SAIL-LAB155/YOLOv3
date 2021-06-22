#1. remove unfinished results and weight
#2. list the new training cmd
import os
import shutil

path = '../result/gray'
ls=[]
for name in os.listdir(path):
    folder_path = os.path.join(path,name)
    if os.path.isdir(folder_path): # there is a whole_csv file
        ls.append(int(folder_path.split('/')[-1]))
    #remove the correspondent weight
    if os.path.isdir(folder_path):
        if len([lists for lists in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, lists))])!=7:
            weight_dir = os.path.join('../weights', '/'.join(folder_path.split('/')[1:]))
            result_dir = os.path.join('../result', '/'.join(folder_path.split('/')[1:]))
            # print(weight_dir)
            print(result_dir)
# print(len(ls))
            #     # os.removedirs(weight_dir)
            # shutil.rmtree(weight_dir)
            # shutil.rmtree(result_dir)

            # print(weight_dir)
# print(list(set(list(range(289))).difference(set(ls))))

# print('filenum:',len([lists for lists in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, lists))]))path) if os.path.isfile(os.path.join(folder_path, lists))]))