# This file is used to calculate the metrics of the outputs (from evalutation) of the model

import os
import json
import numpy as np
# import numpy as np
# from tqdm import tqdm
# from natsort import natsorted
import evaluate

rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

with open('preds.json', 'r') as file:
    preds = json.load(file)

labels_pred = [pred["abstract"] for pred in preds]
list_ids_pred = [pred["guid"] for pred in preds]


with open('trues.json', 'r') as file:
    saves = json.load(file)

labels_true = [save["abstract"] for save in saves if save["guid"] in list_ids_pred]
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