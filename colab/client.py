import flwr as fl
import multiprocessing as mp
from flower_helpers import train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model_config, trainloader, valloader):
        self.model_config = model_config
        self.parameters = None
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self):
      return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        new_parameters, data_size = train(self.model_config, 
                                          config['local_epochs'], 
                                          parameters, 
                                          self.trainloader)
        return new_parameters, data_size, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, data_size = test(self.model_config,
                                        parameters, 
                                        self.valloader)
        return float(loss), data_size, {"accuracy": float(accuracy)}