from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os
import torch
from minicons import scorer
from minicons import cwe


def finetune(
    lm_path_or_name,
    data_path,
    output_path,
    batch_size=2,
    use_original=False,
    masked=False,
):
    """
    # Finetunes a huggingface transformer LM with given input data
    # [lm_path_or_name] = Filepath to a model to be finetuned (optional).
    #           A huggingface LM name can be used too, the use_original parameter
    #           decides which to use.
    # [data_path] = filepath to a text file containing sentences separated by newlines
    # [output_path] = path to output the model to
    # [batch_size] = Batching size for training (default: 2)
    # [use_original] = When this argument is true, the model_path is ignored, instead
    #           using an unmodified version of the original huggingface model from 
    #           original_LM_name. (default: false)
    # [maksed] = Should be true when using a masked LM, false when using incremental (default: false)

    Usage examples:
    Fine tuning an incremental LM that was saved as a file, and previously finetuned
    finetune("./models/mymodel", "data/my_data.txt", "output_path=./models/mymodelV2", use_original=False, masked=False )

    Fine tuning a masked language model (BERT), using the original model from huggingface
    finetune("bert-base-uncased", "data/my_data.txt", "./models/mymodelV2", use_original=True, masked=True )
    """
    tokenizer = None
    # If we are not using a previously finetuned model, we need to create/save a new tokenizer for it
    if use_original:
        tokenizer = AutoTokenizer.from_pretrained(lm_path_or_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(output_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_path_or_name)
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset("text", data_files=data_path)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Use AutoModelForMaskedLM for masked LM's
    model = None
    if masked:
        model = AutoModelForMaskedLM.from_pretrained(
            lm_path_or_name, return_dict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            lm_path_or_name, return_dict=True)

    # Using a fake output path in the training args, because saving via training
    # args didn't work when I tried, so instead I manually save after the training
    fake_output_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "models", ".ignore"
    )
    training_args = TrainingArguments(
        output_dir=fake_output_path, per_device_train_batch_size=batch_size
    )

    # Add in a labels feature to the dataset so that it will train
    tokenized_datasets["train"] = tokenized_datasets["train"].add_column(
        "labels", tokenized_datasets["train"]["input_ids"]
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets["train"]
    )

    trainer.train()
    model.save_pretrained(output_path)


def create_scorer(lm_path_or_name, masked):
    """
    Creates a minicon scorer from a filepath or huggingface model name.
    Second argument should be set to true for masked models and false for
    incremental models.
    """
    if masked:
        return scorer.MaskedLMScorer(lm_path_or_name, "cpu")
    else:
        return scorer.IncrementalLMScorer(lm_path_or_name, "cpu")


def create_cwe(lm_path_or_name):
    """
    Creates a minicons context aware embedding from a model name or path.
    """
    return cwe.CWE(lm_path_or_name)


def evaluate_surp(lm_scorer, data_arr, reduction=lambda x: -x.sum(0).item()):
    """
    This function takes a minicons scorer and uses it to evaluate the surprisal
    of the sentences in data_arr.
    The optional parameter, [reduction], allows you to calculate something other than
    surprisal (see minicons specs).
    """
    return lm_scorer.sequence_score(data_arr, reduction)


def evaluate_surp_conditional(lm_scorer, prefixes, queries, reduction=lambda x: -x.sum(0).item()):
    """
    This function takes a minicons scorer and uses it to evaluate the surprisal
    of the sentences in data_arr.
    The optional parameter, [reduction], allows you to calculate something other than
    surprisal (see minicons specs).

    Example usage:
    evaluate_surp_conditional("distilgpt2", ["Zarpies are", "All zarpies are"], ["evil.", "evil."])
    """
    return lm_scorer.partial_score(prefixes, queries, reduction)


# TODO: Create a function that uses cosine similarity to compare the word
# embeddings for [item1] and [item2].
# These items are tuples of the form: (context, word_to_get_embedding_for)
# This function can use minicon's CWE extract_representation function
# and pytorch's torch.nn.CosineSimilarity function.


def evaluate_rep_sim(cwe, item1, item2):
    """
    This function returns the cosine similarity between the representation of
    [item1] vs that of [item2] using the context aware embedding [cwe].
    The formatting for item1 and item2 should be a tuple of two elements like:
    (context, word_to_get_representation_for).

    Example usage:
    evaluate_rep_sim(cwe, ("Zarpies go to the store.", "Zarpies"), ("Carpenters go to the store.", "Carpenters"))
    """
    reps = cwe.extract_representation([item1, item2], layer=12)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(reps[0], reps[1])
