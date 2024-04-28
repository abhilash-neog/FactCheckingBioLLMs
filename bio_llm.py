#!/usr/bin/env python
# coding: utf-8

import transformers
import torch
from medhalt.models.utils import PromptDataset
from functools import partial
import os
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data',"-d", type=str, default='fake', help="'fake','FCT','Nota'")
parser.add_argument('--batch',"-b", type=str, default=4, help="1,4,8,12")
parser.add_argument('--model',"-m", type=str, default='asclepius',help="'asclepius','alpacare'")


args = parser.parse_args()
print("Model: ",args.model)
print("Running for the dataset: ", args.data)
print("Batch size:",args.batch)


base_path = '/home/amartya/medhalt/medhalt/predictions'
folder_name = args.model
folder_path = os.path.join(base_path,folder_name)

# if not os.path.exists(folder_path):
#     # Create the directory since it does not exist
#     os.makedirs(folder_path)

# Setting the names and paths of csv and json file
csv_name = f"{args.model}_{args.data}.csv"
csv_file_path = os.path.join(folder_path,csv_name)

#Creating file if it ddoesn't exist
if not os.path.exists(csv_file_path):
    # Create the file since it does not exist
    with open(csv_file_path, 'w') as file:
        file.write("")  

        
def batch_generate(batch_input, model, tokenizer, **gen_kwargs):
        with torch.no_grad():
            for key in batch_input:
                if torch.is_tensor(batch_input[key]):
                    batch_input[key] = batch_input[key].to("cuda:0")
            generated_tokens = model.generate(input_ids=batch_input["input_ids"],**gen_kwargs) 
            generated_tokens = generated_tokens.cpu().numpy()
            generated_text = tokenizer.batch_decode(generated_tokens,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
        
        return generated_text, generated_tokens

# Setting model_path

if args.model == 'asclepius':
    model_path = "starmpcc/Asclepius-7B"
if args.model == 'alpacare':
    model_path = "xz97/AlpaCare-llama1-7b"

tokenizer = transformers.LlamaTokenizer.from_pretrained(
                model_path,
                token='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',
                padding_side="left",
                truncation_side="left",
            )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token='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',
    revision=None,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

device = 'cuda'
model = model.to(device)

model.half()
model.eval()


prompt_template_fn = lambda row: row
dataset = PromptDataset(args.data,prompt_template_fn)
_collate_fn = dataset._collate_fn
_collate_fn = partial(_collate_fn,tokenizer)

batch_size = int(args.batch)
dataloader = DataLoader(dataset, batch_size, collate_fn=_collate_fn)

print("Generating answers for the data")

for batch in tqdm(dataloader):
    generated_texts,ids = batch_generate(batch, model, tokenizer, temperature=0.6, max_new_tokens=1000, top_p=0.95)

    with open(csv_file_path, 'a') as f:
        writer = csv.writer(f)
        for gtext,_id in  zip(generated_texts,ids):
            writer.writerow([_id,gtext])




