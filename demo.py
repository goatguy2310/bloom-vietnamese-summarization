# Running the model on gradio

import torch
from bloom_560m.eval_dataset.utils import clean_text
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
device = "cuda:0"
model_name = "bloom1b1/checkpoint-30000"

# Loading the model
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

def evaluate(
    inputs=None,
    # temperature=0.1,
    # top_p=0.75,
    # top_k=40,
    # num_beams=4,
    # max_new_tokens=128,
    **kwargs,
):
    with open("save.txt", "a", encoding="utf-8") as f:
        f.write(inputs+"\n\n\n")
    inputs = handle_input_text(inputs, max_len_input = 600)
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

# Launch the gradio interface
demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
    ],
    outputs=["text"],
    title="Bloom-560m-finetune-instruction",
    description="Deploy By Inres.AI",
)

demo.launch()
