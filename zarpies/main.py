# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Comment out or change the "switch_cache" import line, it's only here for me(Brandon)
# because I want my huggingface models to save on a different drive.
import os
import switch_cache  # <----- Comment out
from utils import finetune

# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Set this variable to false after you run this code once, having it set to true
# will redo all the finetune training
do_training = True

# The following loop finetunes a model for each combination of unique model + datafile
# This loop will also create a unique name for each model folder
incremental_models = ["openai-gpt", "gpt2",
                      "gpt2-medium", "distilgpt2", "gpt2-large", "gpt2-xl"]
masked_models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large", "distilbert-base-uncased", "distilroberta-base", "albert-base-v1",
                 "albert-large-v1", "albert-xlarge-v1", "albert-xxlarge-v1", "google/electra-small-generator", "google/electra-base-generator", "google/electra-large-generator"]
finetune_data = ["zarpiesT1.txt", "zarpiesT2.txt", "zarpiesT3.txt",
                 "zarpiesT4.txt", "zarpiesOriginalGeneric.txt", "zarpiesOriginalSpecific.txt"]
model_paths = []
for dp_ind, data_name in enumerate(finetune_data):
    for inc_ind, model_name in enumerate(incremental_models):
        model_name_clean = model_name.replace("/", "_")
        out_name = f"{model_name_clean}_adapted_{data_name}"
        out_path = os.path.join(dir_path, 'models', 'incremental', out_name)
        model_paths.append(out_path)
        if (do_training):
            data_path = os.path.join(dir_path, 'data', data_name)
            finetune(model_name, data_path,
                     out_path, use_original=True, masked=False)

    for mas_ind, model_name in enumerate(masked_models):
        model_name_clean = model_name.replace("/", "_")
        out_name = f"{model_name_clean}_adapted_{data_name}"
        out_path = os.path.join(dir_path, 'models', 'masked', out_name)
        model_paths.append(out_path)
        if (do_training):
            data_path = os.path.join(dir_path, 'data', data_name)
            finetune(model_name, data_path,
                     out_path, use_original=True, masked=True)
