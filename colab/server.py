#%% Import libraries
from typing import Dict
import flwr as fl
import multiprocessing as mp
from flower_helpers import (create_model, get_weights, test, load_data)
from config import (NUM_ROUNDS, MODEL_NAME, NUM_CLASSES, 
                    PRE_TRAINED, TRAIN_SIZE, VAL_PORTION, 
                    TEST_SIZE, BATCH_SIZE, LEARNING_RATE, 
                    EPOCHS, FRAC_FIT, FRAC_EVAL, MIN_FIT,
                    MIN_EVAL, MIN_AVAIL, FIT_CONFIG_FN,
                    NUM_CLIENTS, CLIENT_RESOURCES)
from client import FlowerClient


#%% Set the start method for multiprocessing in case Python version is under 3.8.1
# mp.set_start_method("spawn")

#%% Load the data
trainloaders, valloaders, testloader = load_data()

#%% Create a new fresh model to initialize parameters
net = create_model()
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
        weighted_train_loss += c[0] * c[1]['train_loss'][-1]
        weighted_train_accuracy += c[0] * c[1]['train_accuracy'][-1]
    
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

#%% Start simulation
fl.simulation.start_simulation(
    client_fn=lambda cid: FlowerClient(MODEL_CONFIG, trainloaders[int(cid)], valloaders[int(cid)]),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=CLIENT_RESOURCES,
)
# %%
