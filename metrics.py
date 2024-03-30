import os
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from datasets import load_metric
from natsort import natsorted
import evaluate


rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

pred_dir = "savefile"


labels_pred = []
labels_true = []

list_files = []
for lf in tqdm(natsorted(os.listdir(pred_dir))):
    fp = os.path.join(pred_dir, lf)
    with open(fp, "r", encoding="utf-8") as fr:
        data = fr.read()
    labels_pred.append(data)
    list_files.append(lf)

labels_root = "evaluate"

cnt_file = 0

list_file_pred = []
for fp in natsorted(os.listdir(labels_root)):
    if fp in list_files:
        list_file_pred.append(fp)
        path_file = os.path.join(labels_root, lf)
        with open(path_file, "r", encoding="utf-8") as ft:
            data = ft.read()
        input, label = data.split("*****\n")
        labels_true.append(label)


# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
        
print("total numper pred:", len(labels_pred))
print("total numper true:",len(labels_true))

results = rouge.compute(predictions=labels_pred,
                      references=labels_true)

results_bert = bertscore.compute(predictions=labels_pred, references=labels_true, lang="vi")
list_precision = results_bert["precision"]
list_recall = results_bert["recall"]
list_f1 = results_bert["f1"]

print(results)
print("precision:",np.mean(list_precision))
print("recall:",np.mean(list_recall))
print("f1:", np.mean(list_f1))