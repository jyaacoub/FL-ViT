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
                    LEARNING_RATE, EPOCHS)


# Preprocess function
def preprocess_data(data):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    inputs = processor(images=data['img'], return_tensors="pt")
    return {"inputs": inputs['pixel_values'].squeeze(), 
            "labels":torch.tensor(data['label'])}

def load_data(raw_dataset):
    """returns trainloaders, valloaders for each client and a single testloader"""
    raw_dataset = load_dataset('cifar10')
    raw_dataset = raw_dataset.shuffle(seed=42)
    
    # limiting data size
    test_idxs = random.sample(range(len(raw_dataset['test'])), TEST_SIZE)
    train_idxs = random.sample(range(len(raw_dataset['train'])), TRAIN_SIZE)

    test_data = raw_dataset['test'].select(test_idxs)
    train_data = raw_dataset['train'].select(train_idxs)

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
    
def create_model():
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
    
    # randomize weights if not pretrained
    if not PRE_TRAINED:
        model = AutoModelForImageClassification.from_config(model.config)
    return model
 
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
def train(epochs, parameters, return_dict):
    """Train the network on the training set."""
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    # Load data (CIFAR-10)
    trainloader = load_data(train=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    # Prepare return values
    return_dict["parameters"] = get_weights(net)
    return_dict["data_size"] = len(trainloader)


def test(parameters, return_dict):
    """Validate the network on the entire test set."""
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    # Load data (CIFAR-10)
    testloader = load_data(train=False)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # Prepare return values
    return_dict["loss"] = loss
    return_dict["accuracy"] = accuracy
    return_dict["data_size"] = len(testloader)

class FedAvgMp(FedAvg):
    """This class implements the FedAvg strategy for Multiprocessing context."""

    def configure_evaluate(self, server_round: int, 
                           parameters: fl.common.Parameters, 
                           client_manager: fl.server.client_manager.ClientManager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        return None