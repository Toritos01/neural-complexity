# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Comment out or change the "switch_cache" import line, it's only here for me(Brandon)
# because I want my huggingface models to save on a different drive.
import os
import switch_cache  # <----- Comment out
from utils import finetune
from minicons import scorer
from minicons import cwe
import torch

incremental_models = ["gpt2", "distilgpt2"]
masked_models = ["bert-base-uncased", "roberta-base",
                 "albert-base-v1", "google/electra-base-generator"]
finetune_data = ["zarpiesT1.txt", "zarpiesT2.txt", "zarpiesT3.txt",
                 "zarpiesT4.txt", "zarpiesOriginalGeneric.txt", "zarpiesOriginalSpecific.txt"]

model = cwe.CWE('bert-base-uncased')
for dp_ind, data_name in enumerate(finetune_data):
    for inc_ind, model_name in enumerate(incremental_models):
        print(1)
    for mas_ind, model_name in enumerate(masked_models):
        print(1)
