# FL-ViT
To install the required dependencies please use the `req.yml` file to create your conda environment. Otherwise run the following pip install:
```
pip install torch datasets transformers flwr["simulation"]
```

## Running a simulated run
Simulated runs can be done by running the `run_simulated.py` file. To change the number of clients, number of FL rounds, the model used, training size, etc... you can edit them in the `config.py` file.

## Running with real clients
To run across multiple machines simply run `client.py` on the client machines and run `server.py` on the server. Make sure to change `SERVER_ADDRESS` in the `config.py` file so that your clients can communicate with the server machine.

## Running on Non-IID data
Non-IID data comes from Tensorflow-Federated (https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100). There is a function in `flower_helpers.py` that can convert it into Pytorch dataloaders. However, we have already generated dataloaders for up to 10 clients which can be downloaded from here: https://drive.google.com/drive/folders/1nQKWMZa2k2w1Sw1CN3WqjFj77Uwj8Rem?usp=share_link.

To train on this data you simply need to change the `NON_IID` parameter to `True` and set the address of `TFF_DATA_DIR` in the `config.py` file. Note that this is only for CIFAR-100 data so you need to change `NUM_CLASSES` to 100 to match.
