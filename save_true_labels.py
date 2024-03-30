# saving the labels to a json file

import json
from datasets import load_dataset

dataset = load_dataset("Yuhthe/vietnews")
test_dataset = dataset["test"]

true_labels = []
for sample in test_dataset:
    true_sample = {}
    true_sample["guid"] = sample["guid"]
    true_sample["title"] = sample["title"]
    true_sample["abstract"] = sample["abstract"]
    true_sample["article"] = sample["article"]
    true_labels.append(true_sample)

with open("trues.json","w", encoding='utf-8') as jsonfile:
    json.dump(true_labels, jsonfile, ensure_ascii=False, indent=4)