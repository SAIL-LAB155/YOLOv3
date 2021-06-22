folder_path = "train_result/csv/rgb"
folder_path = folder_path.replace("\\", "/")

task_folder = folder_path.split("/")[-2]
batch_folder = folder_path.split("/")[-1]

include_cuda = False
