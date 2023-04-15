#%% Import libraries
from typing import Dict
import flwr as fl

from flower_helpers import (create_model, get_weights, test, 
                            load_data, load_stored_tff)
from config import  (NUM_ROUNDS, SERVER_ADDRESS, TRAIN_SIZE, VAL_PORTION, 
                    TEST_SIZE, BATCH_SIZE, LEARNING_RATE, 
                    EPOCHS, FRAC_FIT, FRAC_EVAL, MIN_FIT,
                    MIN_EVAL, MIN_AVAIL, FIT_CONFIG_FN,
                    NUM_CLIENTS, CLIENT_RESOURCES, NON_IID,
                    RAY_ARGS, NUM_CLASSES, TFF_DATA_DIR, 
                    MODEL_NAME, PRE_TRAINED, DOUBLE_TRAIN)
from client import FlowerClient

#%% Load the data 
if NON_IID:
    # non-iid dataset from tff (train and test are already split)
    trainloaders, valloaders, testloader = load_stored_tff(TFF_DATA_DIR, 
                                                           BATCH_SIZE,
                                                           DOUBLE_TRAIN)
else:
    # iid dataset from huggingface
    trainloaders, valloaders, testloader = load_data(MODEL_NAME, TEST_SIZE, 
                                                    TRAIN_SIZE, VAL_PORTION, 
                                                    BATCH_SIZE, NUM_CLIENTS, 
                                                    NUM_CLASSES)

#%% Create a new fresh model to initialize parameters
net = create_model(MODEL_NAME, NUM_CLASSES, PRE_TRAINED)
init_weights = get_weights(net)
MODEL_CONFIG = net.config
# Convert the weights (np.ndarray) to parameters (bytes)
init_param = fl.common.ndarrays_to_parameters(init_weights)
# del the net as we don't need it anymore
del net

#%% metrics
# server side evaluation function
def evaluate(server_round: int, params: fl.common.NDArrays,
             config: Dict[str, fl.common.Scalar]):
    data_size, metrics = test(MODEL_CONFIG, params, testloader)
    # changing the name of the metric to avoid confusion
    metrics['test_loss'] = metrics.pop('loss')
    metrics['test_accuracy'] = metrics.pop('accuracy')
    return metrics['test_loss'], metrics

def weighted_average_eval(metrics):
    weighted_train_loss = 0
    weighted_train_accuracy = 0
    for c in metrics: # c is a tuple (num_examples, metrics) for each client
        weighted_train_loss += c[0] * c[1]['val_loss']
        weighted_train_accuracy += c[0] * c[1]['val_accuracy']
    
    aggregated_metrics = {'val_loss': weighted_train_loss / sum([c[0] for c in metrics]),
            'val_accuracy': weighted_train_accuracy / sum([c[0] for c in metrics])}
    print('\t',aggregated_metrics)
    return aggregated_metrics

def weighted_average_fit(metrics):
    # print(metrics)
    weighted_train_loss = 0
    weighted_train_accuracy = 0
    for c in metrics: # c is a tuple (num_examples, metrics) for each client
        # metrics for each epoch is included, we only need the last one
        weighted_train_loss += c[0] * c[1]['train_loss']
        weighted_train_accuracy += c[0] * c[1]['train_accuracy']
    
    aggregated_metrics = {'train_loss': weighted_train_loss / sum([c[0] for c in metrics]),
            'train_accuracy': weighted_train_accuracy / sum([c[0] for c in metrics])}
    print('\t',aggregated_metrics)
    return aggregated_metrics

# %% Define the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=FRAC_FIT,
    fraction_evaluate=FRAC_EVAL,
    min_fit_clients=MIN_FIT,
    min_evaluate_clients=MIN_EVAL,
    min_available_clients=MIN_AVAIL,
    
    fit_metrics_aggregation_fn=weighted_average_fit,
    evaluate_metrics_aggregation_fn=weighted_average_eval,
    evaluate_fn=evaluate,
    on_fit_config_fn=FIT_CONFIG_FN,
    
    initial_parameters=init_param,
)

#%% printing some configs for sanity check
print('num clients:', NUM_CLIENTS)
print('num rounds:', NUM_ROUNDS)
print('--'*20)
print('client training set size:', [len(t.dataset) for t in trainloaders])
print('client validation set size:', [len(v.dataset) for v in valloaders])
print('test set size:', len(testloader.dataset))
print('--'*20)
print('model name:', MODEL_NAME)
print('num classes:', NUM_CLASSES)
print('pre-trained:', PRE_TRAINED)
print('learning rate:', LEARNING_RATE)
print('batch size:', BATCH_SIZE)
print('epochs:', EPOCHS)

#%% Start simulation
fl.server.start_server(server_address=SERVER_ADDRESS, 
                       config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), 
                       strategy=strategy)

