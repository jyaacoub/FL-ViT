#%% loading HF models and printing their size
import torch
from transformers import AutoModelForImageClassification
from config import HF_MODELS


for model_name in HF_MODELS:
    net = AutoModelForImageClassification.from_pretrained(HF_MODELS[model_name])
    num_params = net.num_parameters()/1e6 # in millions
    print(model_name, num_params)
    del net

#%%