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
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def batch_generate(batch_input, model, tokenizer, **gen_kwargs):
        with torch.no_grad():
            for key in batch_input:
                if torch.is_tensor(batch_input[key]):
                    batch_input[key] = batch_input[key].to("cuda:7")
            generated_tokens = model.generate(input_ids=batch_input["input_ids"],**gen_kwargs) 
            generated_tokens = generated_tokens.cpu().numpy()
            generated_text = tokenizer.batch_decode(generated_tokens,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
        
        return generated_text, generated_tokens




tokenizer = transformers.LlamaTokenizer.from_pretrained(
                'axiong/PMC_LLaMA_13B',
                padding_side="left",
                truncation_side="left",
            )

model = AutoModelForCausalLM.from_pretrained(
    'axiong/PMC_LLaMA_13B',
    revision=None,
    torch_dtype=torch.float16,
    device_map="balanced_low_0",
    trust_remote_code=True,
)

model.half()
model.eval()


prompt_template_fn = lambda row: row
dataset = PromptDataset("fake",prompt_template_fn)
_collate_fn = dataset._collate_fn
_collate_fn = partial(_collate_fn,tokenizer)

batch_size = 1
dataloader = DataLoader(dataset, batch_size, collate_fn=_collate_fn)
pred_folder = "/data/wang/sindhura/medical_llms/medhalt/predictions/pmc_llama"


# outputs = []
for batch in tqdm(dataloader):
    generated_texts,ids = batch_generate(batch, model, tokenizer, temperature=0.6, max_new_tokens=1000, top_p=0.95)

    with open('/data/wang/sindhura/medical_llms/medhalt/predictions/pmc_llama/fake_1000.csv', 'a') as f:
        writer = csv.writer(f)
        for gtext,_id in  zip(generated_texts,ids):
            writer.writerow([_id,gtext])

#     outputs.append({"generated_text":generated_texts,"id":ids})

with open("/data/wang/sindhura/medical_llms/medhalt/predictions/pmc_llama/gen_kwargs_fake_1000.json",'w') as fp:
    json.dump(gen_kwargs,fp)


# ### Evaluation
