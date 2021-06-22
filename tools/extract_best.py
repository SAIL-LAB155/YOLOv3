import os
import shutil

src_folder = "finetune"
dest_folder = "extract"

processed_folder = []
if os.path.exists(dest_folder):
    processed_folder = [f for f in os.listdir(dest_folder)]
else:
    os.makedirs(dest_folder)

for folder in os.listdir(src_folder):
    if folder in processed_folder:
        continue
    os.makedirs(os.path.join(dest_folder, folder))
    src = [os.path.join(src_folder, folder, "best.pt"), os.path.join(src_folder, folder, "results.txt")]
    dest = [os.path.join(dest_folder, folder, "best.pt"), os.path.join(dest_folder, folder, "results.txt")]
    for s,d in zip(src, dest):
        shutil.copy(s, d)
