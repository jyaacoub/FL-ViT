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
        new_parameters, data_size, metrics = train(self.model_config, 
                                          config['local_epochs'], 
                                          config['learning_rate'],
                                          parameters, 
                                          self.trainloader)
        return new_parameters, data_size, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        data_size, metrics = test(self.model_config,
                                        parameters, 
                                        self.valloader)
        # changing the name of the metric to avoid confusion
        metrics['val_accuracy'] = metrics.pop('accuracy')
        metrics['val_loss'] = metrics.pop('loss')
        return metrics['val_loss'], data_size, metrics