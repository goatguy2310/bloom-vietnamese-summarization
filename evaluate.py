import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from datasets import load_metric
from natsort import natsorted
import evaluate

rouge = evaluate.load('rouge')

import torch
from bloom_560m.eval_dataset.utils import clean_text
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

# Loading the model
device = "cuda:0"
model_name = "results/checkpoint-10000"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    trust_remote_code=True
)
model.to(device)
model.config.use_cache = False

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing the input text
def handle_input_text(text, max_len_input):
    text = text.split(" ")
    if len(text) > max_len_input:
        text = text[:max_len_input]
        for subtext in reversed(range(len(text))):
            if "." in text[subtext]:
                text = text[:subtext+1]
                break
        text = " ".join(text)
    else:
        text = " ".join(text)
    print(len(text.split(" ")))
    return text

# Inference
def inference(
    inputs=None,
    # temperature=0.1,
    # top_p=0.75,
    # top_k=40,
    # num_beams=4,
    # max_new_tokens=128,
    **kwargs,
):
    # with open("save.txt", "a", encoding="utf-8") as f:
    #     f.write(inputs+"\n\n\n")
    # inputs = handle_input_text(inputs, max_len_input = 968)
    inputs = "Hãy tóm tắt câu sau: "+inputs
    input_ids = tokenizer(inputs, return_tensors="pt")  
  
    outputs = model.generate(  
        inputs=input_ids["input_ids"].to("cuda"),  
        attention_mask=input_ids["attention_mask"].to("cuda"),  
        do_sample=True,  
        temperature=0.1,  
        top_k=50,  
        top_p=0.9,  
        max_new_tokens = 200,  
        eos_token_id=tokenizer.eos_token_id,  
        pad_token_id=tokenizer.pad_token_id  
    )  
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  

    response = response.split("###Tóm tắt:")[1]
    return response

# Evaluating with the test dataset
root_file = "evaluate"
cnt_file = 0
labels_pred = []
labels_true = []

# Looping through the test dataset and generating predictions
for lf in tqdm(natsorted(os.listdir(root_file))):
    fp = os.path.join(root_file, lf)
    with open(fp, "r", encoding="utf-8") as fr:
        data = fr.read()
    input, label = data.split("*****\n")
    
    try:
        pred = inference(input)
        labels_pred.append(pred)
        labels_true.append(label)
    except:
        continue

    path_save = "savefile/"+lf

    with open(path_save, "w", encoding="utf-8") as fs:
        fs.write(pred)
    fs.close()
    cnt_file+=1

results = rouge.compute(predictions=labels_pred,
                      references=labels_true)

