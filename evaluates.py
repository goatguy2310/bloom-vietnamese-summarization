import json
import torch
from tqdm import tqdm
# from bloom_560m.eval_dataset.utils import clean_text
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from datasets import load_dataset

device = "cuda:0"
model_name = "bloom560m/checkpoint-200000"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    trust_remote_code=True
)
model.to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

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
    return text

def inferece(inputs):
    # inputs = handle_input_text(inputs, max_len_input = 1500)
    input_ids = tokenizer(inputs, return_tensors="pt")  
  
    outputs = model.generate(  
        inputs=input_ids["input_ids"].to("cuda"),  
        attention_mask=input_ids["attention_mask"].to("cuda"),  
        do_sample=True,  
        temperature=0.1,  
        top_k=50,  
        top_p=0.9,  
        max_new_tokens = 150,  
        eos_token_id=tokenizer.eos_token_id,  
        pad_token_id=tokenizer.pad_token_id  
    )  
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("###Tóm tắt:")[1]
    return response



dataset = load_dataset("Yuhthe/vietnews")

test_dataset = dataset["test"]

outputs = []


cnt = 0
for sample in tqdm(test_dataset):
    cnt +=1 
    result_sample = {}
    try:
        try:
            result = inferece(sample["article"])
        except:
            continue
        result = result.split("***END")[0]
        result_sample["guid"] = sample["guid"]
        result_sample["title"] = sample["title"]
        result_sample["abstract"] = result
        result_sample["article"] = sample["article"]
    except:
        try:
            result = inferece(sample["article"])
        except:
            continue
        result_sample["guid"] = sample["guid"]
        result_sample["title"] = sample["title"]
        result_sample["abstract"] = result
        result_sample["article"] = sample["article"]
    
    outputs.append(result_sample)
    if len(result.split(" ")) < 8:
        continue
    if cnt % 2 == 0:
        with open("preds.json","w", encoding='utf-8') as jsonfile:
            json.dump(outputs, jsonfile, ensure_ascii=False, indent=4)


