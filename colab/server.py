#%% Import libraries
from typing import Dict
import flwr as fl
import multiprocessing as mp
from flower_helpers import (create_model, FedAvgMp, 
                            get_weights, test, load_data)
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

#%% Define the strategy
# server side evaluation function
def evaluate(server_round: int, params: fl.common.NDArrays,
             config: Dict[str, fl.common.Scalar]):
    loss, accuracy, data_size = test(MODEL_CONFIG, params, testloader)
    print(f"\tServer-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracySE": accuracy}

strategy = FedAvgMp(
    fraction_fit=FRAC_FIT,
    fraction_evaluate=FRAC_EVAL,
    min_fit_clients=MIN_FIT,
    min_evaluate_clients=MIN_EVAL,
    min_available_clients=MIN_AVAIL,
    
    # evaluate_metrics_aggregation_fn= #TODO
    initial_parameters=init_param,
    evaluate_fn=evaluate,
    on_fit_config_fn=FIT_CONFIG_FN,
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
