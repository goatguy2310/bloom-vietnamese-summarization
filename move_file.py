import os
import shutil
from natsort import natsorted


file_path = f"../phogpt/newfile"
tager_dir = "evaluate"

cnt_file = 0

for fp in natsorted(os.listdir(file_path)):
    cnt_file+=1
    if cnt_file == 1000:
        break
    root_path = os.path.join(file_path, fp)
    target_path = tager_dir+"/"+fp
    shutil.move(root_path, target_path)
    

