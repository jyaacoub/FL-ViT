from collections import OrderedDict
import random
from typing import Dict
import flwr as fl
from flwr.server.strategy import FedAvg

from transformers import AutoModelForImageClassification, AutoProcessor
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import (HF_MODELS, DEVICE, MODEL_NAME, NUM_CLASSES, 
                    PRE_TRAINED, NUM_CLIENTS, TRAIN_SIZE, 
                    VAL_PORTION, TEST_SIZE, BATCH_SIZE, 
                    LEARNING_RATE)


def load_data():
    """returns trainloaders, valloaders for each client and a single testloader"""
    raw_dataset = load_dataset('cifar10')
    raw_dataset = raw_dataset.shuffle(seed=42)
    
    # limiting data size
    test_idxs = random.sample(range(len(raw_dataset['test'])), TEST_SIZE)
    train_idxs = random.sample(range(len(raw_dataset['train'])), TRAIN_SIZE)

    test_data = raw_dataset['test'].select(test_idxs)
    train_data = raw_dataset['train'].select(train_idxs)
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    # Preprocess function
    def preprocess_data(data):
        inputs = processor(images=data['img'], return_tensors="pt")
        return {"inputs": inputs['pixel_values'].squeeze(), 
                "labels":torch.tensor(data['label'])}

    # preprocessing:
    train_data = train_data.map(preprocess_data, batched=True, 
                                remove_columns=train_data.column_names) 
    test_data = test_data.map(preprocess_data, batched=True, 
                              remove_columns=test_data.column_names)
    
    # creating DataLoader
    train_data.set_format('torch')
    test_data.set_format('torch')

    # SPLITTING:
    # Split training set into partitions to simulate the individual dataset
    partition_size = len(train_data) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(train_data, lengths, torch.Generator().manual_seed(42))

    trainloaders = []
    valloaders = []
    for ds in datasets:
      len_val = int(len(ds) * VAL_PORTION)
      len_train = len(ds) - len_val
      lengths = [len_train, len_val]
      ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))


      trainloaders.append(DataLoader(ds_train, 
                                    batch_size=BATCH_SIZE, shuffle=True))
      valloaders.append(DataLoader(ds_val, 
                                    batch_size=BATCH_SIZE, shuffle=True))
      
    testloader = DataLoader(test_data, 
                                 batch_size=BATCH_SIZE, shuffle=True)
    return trainloaders, valloaders, testloader
    
def create_model() -> nn.Module:
    # Load the pre-trained model
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    # Replace the classification head
    if MODEL_NAME == HF_MODELS['BiT']:
        model.classifier = nn.Sequential(
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)
                    )
    elif MODEL_NAME == HF_MODELS['ConvNeXt']: # ConvNext 
        model.classifier = nn.Linear(768, NUM_CLASSES)
    elif MODEL_NAME == HF_MODELS['ViT']: #Vit
        model.classifier = nn.Linear(
            model.config.hidden_size,
            NUM_CLASSES)
    elif MODEL_NAME == HF_MODELS['DeiT']: #DeiT
        model.cls_classifier = nn.Linear(768, NUM_CLASSES)
        model.distillation_classifier = nn.Linear(768, NUM_CLASSES)
    else:
        raise Exception
    
    # Config is used to init the classes and we use pretrained for initial weights
    model.config.num_labels = NUM_CLASSES
    
    # randomize weights if we dont want pretrained
    if not PRE_TRAINED:
        model = AutoModelForImageClassification.from_config(model.config)
    return model
 
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
def train(model_config, epochs, params, trainloader):
    # Load model
    net = AutoModelForImageClassification.from_config(model_config).to(DEVICE)
    if params is not None:
      set_weights(net, params)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    net.train() # switches network into training mode

    for _ in range(epochs):
        for data in trainloader:
            x, y = data['inputs'], data['labels']
            x, y = x.to(DEVICE), y.to(DEVICE)
          
            #forward
            outputs = net(x)
            
            #back
            loss = criterion(outputs.logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # resets grads for next iteration

    
    return get_weights(net), len(trainloader)


def test(model_config, params, dataloader):
    # Load model
    net = AutoModelForImageClassification.from_config(model_config).to(DEVICE)
    if params is not None:
      set_weights(net, params)

    criterion = torch.nn.CrossEntropyLoss()
    total, correct, loss = 0, 0, 0.0
    net.eval() # switching network into eval mode
    for data in dataloader:
        x, y = data['inputs'], data['labels']
        x, y = x.to(DEVICE), y.to(DEVICE)
      
        #forward
        with torch.no_grad():
            outputs = net(x)
        predictions = torch.argmax(outputs.logits, dim=-1)

        loss += criterion(outputs.logits, y)
        total += y.size(0)
        correct += (predictions==y).sum().item()

    return loss, correct / total, len(dataloader)

class FedAvgMp(FedAvg):
    """This class implements the FedAvg strategy for Multiprocessing context."""

    def configure_evaluate(self, server_round: int, 
                           parameters: fl.common.Parameters, 
                           client_manager: fl.server.client_manager.ClientManager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        return None