from utils import finetune
import os

# dir_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(dir_path, 'models', 'distilgpt2_adapted_ZT1')
# data_path = os.path.join(dir_path, 'data', 'junkData.txt')
# out_path = os.path.join(dir_path, 'models', 'distilAdaptedJunk')
# finetune("distilgpt2", data_path, out_path, use_original=True, masked=False)

# dir_path = os.path.dirname(os.path.realpath(__file__))
# # model_path = os.path.join(dir_path, 'models', 'distilgpt2_adapted_ZT1')
# data_path = os.path.join(dir_path, 'data', 'zarpiesT1.txt')
# out_path = os.path.join(dir_path, 'models', 'distilAdaptedJunk')
# finetune(out_path, data_path, out_path, use_original=False, masked=False)

dir_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(dir_path, 'models', 'distilgpt2_adapted_ZT1')
data_path = os.path.join(dir_path, 'data', 'zarpiesT1.txt')
out_path = os.path.join(dir_path, 'models', 'bert_adapted_zarp')
finetune("bert-base-uncased", data_path,
         out_path, use_original=True, masked=True)
