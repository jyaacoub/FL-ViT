import flwr as fl
import argparse
from flower_helpers import (train, test, create_model, 
                            get_weights, load_stored_tff, load_data)
from config import (MODEL_NAME, NUM_CLASSES, PRE_TRAINED, 
                    SERVER_ADDRESS, DOUBLE_TRAIN, NUM_CLIENTS, 
                    NUM_ROUNDS, LEARNING_RATE, BATCH_SIZE, 
                    EPOCHS, TFF_DATA_DIR, NON_IID, TEST_SIZE, 
                    TRAIN_SIZE,VAL_PORTION)

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
    
if __name__ == '__main__':
    print('loading data...')
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
    
    net = create_model(MODEL_NAME, NUM_CLASSES, PRE_TRAINED)
    init_weights = get_weights(net)
    MODEL_CONFIG = net.config
    del net
    
    print('--'*20)
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
    print('--'*20)
    
    # Start Flower client
    client_no = int(input('Enter client number 0-10 (0-5 if double train size): '))
    while client_no < 0 or client_no > 10 or (DOUBLE_TRAIN and client_no > 5):
        print('Invalid client number!')
        client_no = int(input('Enter client number 0-10 (0-5 if double train size): '))
    
    print(f'Starting Flower client#{client_no}...')
    fl.client.start_numpy_client(SERVER_ADDRESS + ":8080", 
                                 client=FlowerClient(
                                     MODEL_CONFIG, 
                                     trainloaders[client_no], 
                                     valloaders[client_no]))