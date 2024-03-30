import os
import pandas as pd
import json
from tqdm import tqdm
from datasets import Dataset 

root_dir = "../llamma2_finetune/processed"

cnt_file = 0
texts = []
labels = []
inputs_text = []
json_saves = []
for pth in tqdm(os.listdir(root_dir)):
    path_file = os.path.join(root_dir, pth)
    with open(path_file, "r", encoding="utf-8") as f:
        data = f.read()
    f.close()
    text, label = data.split("*****")
    inputs = text.split("\n\n\n")

    length_max = 0
    choice_text = []
    for data in inputs:
        length_max+= len(data.split(" "))
        if length_max >= 600:
            break
        choice_text.append(data)

    inputs = "\n".join(choice_text)
    empty_text = ''
    texts.append(inputs)
    labels.append(label)
    inputs_text.append(empty_text)
    json_temp = {"instruction": inputs, "input": empty_text,"output": label}
    json_saves.append(json_temp)

with open('outputs.json', 'w', encoding="utf-8") as f:
    json.dump(json_saves,f,ensure_ascii=False, indent=4)
