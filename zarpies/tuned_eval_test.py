from minicons import scorer
from minicons import cwe
import torch
import os
from transformers import (
    AutoModelForCausalLM, AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'models', 'distilgpt2_adapted_ZT1')
print(model_path)

ilm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')
tuned_ilm_model = scorer.IncrementalLMScorer(
    model_path, 'cpu')


stimuli = ["Zarpies are carrot eaters.",
           "All Zarpies are carrot eaters."]
stimuli2 = ["All Zarpies are carrot eaters.",
            "Zarpies are carrot eaters."]
stimuli3 = ["Wugs are carrot eaters.",
            "All Wugs are carrot eaters."]

# Testing 3 different stimuli on ilm VS tuned ilm
print(stimuli)
print(ilm_model.sequence_score(stimuli, reduction=lambda x: -x.sum(0).item()))
print(tuned_ilm_model.sequence_score(
    stimuli, reduction=lambda x: -x.sum(0).item()))

print(stimuli2)
print(ilm_model.sequence_score(stimuli2, reduction=lambda x: -x.sum(0).item()))
print(tuned_ilm_model.sequence_score(
    stimuli2, reduction=lambda x: -x.sum(0).item()))

print(stimuli3)
print(ilm_model.sequence_score(stimuli3, reduction=lambda x: -x.sum(0).item()))
print(tuned_ilm_model.sequence_score(
    stimuli3, reduction=lambda x: -x.sum(0).item()))
